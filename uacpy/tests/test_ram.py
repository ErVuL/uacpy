"""RAM parabolic-equation-focused tests.

Pade parameters and stability, the range/depth grids the deck carries, and the
tolerances the Fortran march itself imposes on where a receiver can land.
"""

import subprocess
import re
import warnings
from pathlib import Path

import pytest
import numpy as np

import uacpy
from uacpy.models import RAM, RunMode
from uacpy.models.ram.collins import MAX_BATHY_SECTIONS
from uacpy.models.ram.mpirams import MPIRAMS_RANGE_TOL_M
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
    NumericsWarning,
)
from uacpy import Field
from uacpy.core import (Altimetry, Bathymetry, 
    Environment, Source, Receiver, SoundSpeedProfile, BoundaryProperties,
    Bottom, SeabedColumn, SedimentLayer,
)
from uacpy.core.absorption import ConstantAbsorption, FrancoisGarrison, Thorp
from uacpy.core.acoustics.attenuation import absorption_thorp
from uacpy.tests.conftest import (range_independent_layered_env,
                                  wide_range_dependent_env)
from uacpy.io.mpirams_writer import write_inpe, write_water_attenuation_file
from uacpy.io.ramsurf_writer import write_ramin
from uacpy.models.ram import (
    _domain as ram_domain,
    _band as ram_band,
    _stability as ram_stability,
    grid as ram_grid,
    mpirams as ram_mpirams,
    collins as ram_collins,
)
from uacpy.tests.conftest import recorded_warnings
from uacpy.tests.conftest import make_pekeris

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
        (peramx.f90:74-105, order pinned by ``write_inpe``)."""
        decks = sorted(work_dir.rglob('in.pe'))
        assert decks, f"no in.pe under {work_dir}"
        return decks[0].read_text().splitlines()[6].split()

    @pytest.mark.parametrize('n_pade', [2, 6, 8])
    def test_ram_pade_order(self, ram_env, ram_source, ram_receiver, n_pade,
                            tmp_path):
        """Every supported Padé-coefficient count marches to a finite field.

        ``n_pade`` is the number of terms in the rational approximation of
        the propagator, and each term costs one tridiagonal solve per range
        step. A count the coefficient solver cannot deliver shows up as a
        stopped binary or an all-NaN grid, not as a small accuracy change, so
        finiteness is the discriminating assertion — plus the deck itself,
        which must carry the requested count."""
        ram = RAM(verbose=False, dr=20.0, dz=2.0, n_pade=n_pade,
                  work_dir=tmp_path, cleanup=False)
        result = ram.compute_tl(
            env=ram_env, source=ram_source, receiver=ram_receiver,
        )
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))
        assert int(self._inpe_pade_line(tmp_path)[0]) == n_pade

    def test_ram_stability_parameter(self, ram_env, ram_source, ram_receiver,
                                     tmp_path):
        """``n_stability`` is the number of evanescent-spectrum points at
        which the rational approximation is forced to vanish, trading
        ``2n - ns`` accuracy constraints for stability (RAM manual §2). It
        changes the Padé coefficients, so the march must still complete and
        stay finite — and the deck's ``np nss`` line must carry the count."""
        ram = RAM(verbose=False, dr=20.0, dz=2.0, n_stability=1,
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
        assert float(result.run_settings.engine.grids[0].dr) == pytest.approx(10.0)
        assert float(result.run_settings.engine.grids[0].dz) == pytest.approx(0.5)

    def test_ram_tl_writes_the_one_bin_band_whatever_q_t_hold(
        self, ram_env, ram_source, ram_receiver, monkeypatch,
    ):
        """COHERENT_TL writes ``(Q, T) = (1e6, 1.0)`` to in.pe even when
        ``RAM(q_factor=…, record_duration=…)`` pins a broadband band: the TL field keeps only the
        centre bin, so the pinned band would be marched and discarded.
        """
        captured = {}

        def fake_write_inpe(*args, **kwargs):
            captured['q_factor'] = kwargs['q_factor']
            captured['record_duration'] = kwargs['record_duration']
            raise RuntimeError("stop after writing in.pe")

        monkeypatch.setattr(ram_mpirams, 'write_inpe', fake_write_inpe)

        ram = RAM(q_factor=4.0, record_duration=20.0, dr=20.0, dz=2.0, verbose=False)
        with pytest.raises(RuntimeError, match="stop after writing in.pe"):
            ram.compute_tl(env=ram_env, source=ram_source, receiver=ram_receiver)
        assert captured['q_factor'] == 1e6
        assert captured['record_duration'] == 1.0


class TestEveryBackendRunsItsFamilysSteps:
    """M-22: RAM's stage hooks dispatch on the backend's family through one
    table (``_model._BACKEND_STEPS``), mpiramS's steps and the Collins
    codes' side by side."""

    def test_the_family_table_covers_every_backend(self):
        from uacpy.models.ram import _model
        assert set(_model._FAMILY) == set(RAM._BACKENDS)
        assert set(_model._FAMILY.values()) == set(_model._BACKEND_STEPS)

    @staticmethod
    def _spied(monkeypatch, backend, mode):
        import types
        from uacpy.models.ram import _model
        called = []

        def spy(name, value=None):
            def record(*args, **kwargs):
                called.append(name)
                return value
            return record
        for name in ('write_mpirams_deck', 'write_collins_deck', 'read_psif',
                     'read_collins_output', 'assemble_tl_field',
                     'assemble_collins_tl_field'):
            monkeypatch.setattr(_model, name, spy(name))
        monkeypatch.setattr(_model, 'collins_deck_base', spy('deck', {}))
        monkeypatch.setattr(_model, 'water_alpha_band', spy('alpha', 'a'))
        model = RAM(verbose=False)
        monkeypatch.setattr(model, '_run_binary', spy('_run_binary'))
        monkeypatch.setattr(model, '_run_collins_binary',
                            spy('_run_collins_binary'))
        grid = types.SimpleNamespace(frequency=100.0, dr=10.0, dz=1.0,
                                     zmax=200.0)
        settings = types.SimpleNamespace(
            engine=types.SimpleNamespace(backend=backend, grids=(grid,),
                                         c0=1500.0,
                                         marched_frequencies=(100.0,)),
            mode=mode)
        inputs = types.SimpleNamespace(settings=settings, work_dir=None,
                                       env=None, source=None, receiver=None)
        return model, settings, inputs, called

    @pytest.mark.parametrize('backend, family', [
        ('mpirams', 'mpirams'), ('ramgeo', 'collins'), ('rams', 'collins'),
        ('ramsurf', 'collins')])
    def test_each_hook_runs_its_family_step(self, monkeypatch, backend,
                                            family):
        """Each stage-4/5 hook reaches the writer, binary, reader and
        assembler of the backend's family, and no other family's."""
        model, _, inputs, called = self._spied(monkeypatch, backend,
                                               RunMode.COHERENT_TL)
        model._write_input(inputs)
        model._launch(inputs, None)
        model._read_output(inputs, None)
        model._to_result(inputs, None, None)
        assert called == {
            'mpirams': ['write_mpirams_deck', '_run_binary', 'read_psif',
                        'assemble_tl_field'],
            'collins': ['write_collins_deck', '_run_collins_binary',
                        'read_collins_output', 'assemble_collins_tl_field'],
        }[family]

    @pytest.mark.parametrize('backend, prepared', [
        ('mpirams', None), ('ramgeo', {'water_alpha': 'a'}),
        ('rams', {'water_alpha': 'a'}), ('ramsurf', {'water_alpha': 'a'})])
    def test_only_a_collins_band_prepares_a_shared_deck(
            self, monkeypatch, backend, prepared):
        """A Collins band cuts its deck once for every bin; mpiramS
        marches the band in one launch and prepares nothing."""
        model, settings, _, _ = self._spied(monkeypatch, backend,
                                            RunMode.BROADBAND)
        assert model._prepare_launches(None, settings) == prepared


class TestTheGridConstraintsApplyInOrder:
    """M-22: ``grid.compute_grid_lytaev`` applies the constraints its
    optimiser does not model one function each, in a fixed order."""

    ORDER = ('tighten_rams_dr', 'cap_dr_at_the_collins_output_stride',
             'place_the_seafloor_in_its_cell', 'cap_the_depth_points',
             'floor_dz', 'resolve_the_shear_wavelength',
             'keep_the_source_below_row_one',
             'put_the_ramsurf_surface_on_a_node')

    def test_every_constraint_runs_once_in_order(self, monkeypatch):
        seen = []
        for name in self.ORDER:
            real = getattr(ram_grid, name)

            def spy(*args, _name=name, _real=real, **kwargs):
                seen.append(_name)
                return _real(*args, **kwargs)
            monkeypatch.setattr(ram_grid, name, spy)
        model = RAM(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ram_grid.compute_grid_lytaev(
                _env_flat_fluid(), 100.0, max_range=5000.0, kind='ramsurf',
                zs=50.0, knobs=model._knob_record(), log=model._log,
                speed_bounds=model._speed_bounds)
        assert seen == list(self.ORDER)


class TestAKnobReassignedAfterConstructionReachesTheNextRun:
    """RAM resolves its grid from its knobs as they stand when a run
    starts (``RAM._knob_record``), so a knob reassigned between runs
    reaches the next run as the same value given to the constructor
    does."""

    def test_reassigned_dz_and_dr_match_a_constructor_pin(self):
        env = _env_flat_fluid()
        src = uacpy.Source(depths=50.0, frequencies=100.0)
        rcv = uacpy.Receiver(depths=np.linspace(5.0, 95.0, 10),
                             ranges=np.linspace(100.0, 2000.0, 20))
        # both dz put the 100 m seafloor a quarter cell below a node
        model = RAM(verbose=False, dz=1.99005, dr=20.0)
        before = model.run_settings(env, src, rcv).engine.grids
        model.dz, model.dr = 0.499376, 5.0
        after = model.run_settings(env, src, rcv).engine.grids
        pinned = RAM(verbose=False, dz=0.499376, dr=5.0).run_settings(
            env, src, rcv).engine.grids
        assert (after[0].dz, after[0].dr) == (0.499376, 5.0)
        assert after == pinned
        assert after != before


class TestRAMRangeDependentSSPShortRange:
    """A range-dependent SSP over a SHORT receiver range must not crash mpiramS.

    mpiramS's horizontal-interpolation branch (``horizontal_interpolation=1``) sizes its SSP
    resample grid as ``nrp = nint(maxval(rmax)/10000)``
    (``third_party/mpiramS/src/peramx.f90:253``). That rounds to 0 for any max
    receiver range below 5 km — a zero-length allocation, an all-NaN field and
    a SIGABRT (exit -6) — and to 1 below 15 km, which resamples the whole run
    onto a single profile. uacpy therefore drives mpiramS with ``horizontal_interpolation=0``, so
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
            depths=z, sound_speed=data, ranges=np.array([0.0, 3000.0]),
        )
        return Environment(bathymetry=100.0, ssp=ssp, bottom=self._BOTTOM)

    @pytest.mark.parametrize('rmax', [1500.0, 2000.0, 2500.0, 4000.0])
    def test_short_range_is_finite(self, rmax):
        """Every rmax here is below the 5 km at which ``nint(rmax/10000)``
        first rounds up to 1, so each one hits the zero-length branch if the
        ``horizontal_interpolation=1`` path is ever taken."""
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
        """``horizontal_interpolation=0`` must not collapse range dependence: a varying SSP has to
        give a different short-range field than a range-independent one."""
        src = Source(depths=25.0, frequencies=50.0)
        rcv = Receiver(depths=[50.0], ranges=[1000.0, 2000.0, 3000.0])
        z = np.array([0.0, 100.0])
        ri = Environment(
            bathymetry=100.0,
            ssp=SoundSpeedProfile(depths=z, sound_speed=np.array([[1500.0], [1490.0]])),
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
    env = Environment(name='p', bathymetry=200.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=1800.0, density=1.8,
                                                attenuation=0.5))
    src = Source(depths=50.0, frequencies=100.0)
    rcv = Receiver(depths=100.0, ranges=np.array([1000.0]))

    with recorded_warnings() as w:
        RAM(verbose=False).run(env, src, rcv)
    budget = [x for x in w if 'accuracy budget' in str(x.message)]
    assert not budget, f"default run warned about its own default: {budget}"

    with recorded_warnings() as w:
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
    """``zread`` (``ramsurf1.5.f:194-225``, identical in
    ``ramgeo1.5.f:209-240`` and ``rams0.5.f:212-243``) pins each block point
    to the node ``i = 1.5 + z/dz`` and remembers only the immediately
    preceding index, so its collision push-down at :208 protects one
    duplicate depth and no more. A layer thinner than ``dz/2`` therefore had
    its two faces land in one cell, the deeper value overwrote the
    shallower, and the fill loop at :218-219 ramped linearly across the rest
    of the sub-bottom. Measured on the *default* dispatch path
    (``ram._dispatch.prefer_ramgeo`` routes any layered fluid bottom there):
    a 0.6 m mud layer over an 1800 m/s basement came out as a 692 m gradient
    1500 → 1800 m/s, 22.6 dB from Scooter on the same environment."""

    @staticmethod
    def _env(thickness):
        from uacpy.core.boundary import SedimentLayer
        from uacpy.core.bottom import Bottom, SeabedColumn
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
        *relative to the seafloor*, writes a bare half-space as two breakpoints,
        and has extra attenuation points from the absorbing ramp."""
        env = self._env(thickness)
        src, rcv = self._src_rcv()
        result = RAM(verbose=False, work_dir=str(tmp_path),
                     cleanup=False).run(env, src, rcv)
        dz = float(result.run_settings.engine.grids[0].dz)
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
        ram = np.asarray(RAM(verbose=False).run(env, src, rcv).dB)
        scooter = np.asarray(Scooter(verbose=False, c_low=1400.0,
                                     c_high=1e9).run(env, src, rcv).dB)
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
        assert float(result.run_settings.engine.grids[0].dz) == pytest.approx(dz)

    @pytest.mark.parametrize('dz', [None, 1.886792])
    def test_mpirams_is_exempt(self, dz):
        """Only the three Collins backends pin block points to grid nodes — the
        ``1.5+zi/dz`` arithmetic appears in ``ramsurf1.5.f``, ``ramgeo1.5.f`` and
        ``rams0.5.f`` and in no mpiramS source, which interpolates the profile
        onto the grid with ``interpolators.f90``'s ``interp1``. Tightening or
        refusing an mpiramS run would be a false positive."""
        env = self._env(0.6)
        src, rcv = self._src_rcv()
        result = RAM(verbose=False, backend='mpirams', dz=dz).run(env, src, rcv)
        if dz is not None:
            assert float(result.run_settings.engine.grids[0].dz) == pytest.approx(dz)

    def test_a_bottom_without_layers_is_untouched(self):
        """The cap must not tighten a grid with no block step to resolve."""
        env_plain, env_layered = self._env(None), self._env(0.6)
        src, rcv = self._src_rcv()
        dz_plain = float(RAM(verbose=False).run(env_plain, src, rcv)
                         .run_settings.engine.grids[0].dz)
        dz_layered = float(RAM(verbose=False).run(env_layered, src, rcv)
                           .run_settings.engine.grids[0].dz)
        assert dz_plain > dz_layered
        # A pure half-space is two breakpoints plus the ramp's start, so the
        # block is not empty — what matters is that its points do not collide.
        model = RAM(verbose=False)
        assert not ram_collins.block_loses_a_point(
            env_plain, dz_plain, 800.0, 'ramgeo', 50.0,
            knobs=model._knob_record(), speed_bounds=model._speed_bounds)

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
        assert float(result.run_settings.engine.grids[0].dz) == pytest.approx(0.25)

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
    """``ram.collins.ramp_absorbing_attenuation`` must not overwrite the
    deepest layer.

    Collins' readme (``ramsurf/readme.orig:127-133``) makes the ramp *"a
    buffer region"* below the modelled seabed, separate from the seabed's own
    attenuation. The block carries duplicated abscissae at each interface, so
    the point that pins a layer's value at its own base sits exactly at the
    ramp start whenever the absorber clamps to the sediment base — a strict
    ``<`` dropped it, and ``np.interp`` on the duplicate then returned the
    half-space value, marching a 0.02 dB/lambda layer at up to 5.0."""

    @staticmethod
    def _ramp(pairs, z_sediment_base, z_bottom, width=1000.0, floor=10.0):
        import types
        return ram_collins.ramp_absorbing_attenuation(
            pairs, z_sediment_base, z_bottom, width,
            knobs=types.SimpleNamespace(absorber_attenuation=floor))

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


class TestTheAbsorberIsCountedInReferenceWavelengths:
    """``absorber_width_wavelengths`` counts wavelengths of the PE reference
    speed ``c0`` at every site that reads it. Counting basement wavelengths
    instead was measured and refuted: on lossless rock and hard rock the
    field moves ≤ 0.3 / 0.000 dB between the width and twice it under
    either count, for ×2.1 depth nodes on hard rock at 100 Hz and ×2.3 at
    25 Hz — the ramp already absorbs 37 dB one way on hard rock."""

    FREQ = 100.0

    @staticmethod
    def _half_space(c_bottom):
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=c_bottom, density=2.0,
                                      attenuation=0.1))

    def test_a_hard_rock_basement_gets_reference_wavelengths(self):
        model = RAM(verbose=False)
        env = self._half_space(5500.0)
        c0 = ram_domain.resolve_c0(env, knobs=model._knob_record(),
                                   speed_bounds=model._speed_bounds)
        assert 1500.0 < c0 < 5500.0
        assert ram_domain.absorbing_layer_thickness(
            env, self.FREQ, knobs=model._knob_record(),
            speed_bounds=model._speed_bounds) == pytest.approx(
            model.absorber_width_wavelengths * c0 / self.FREQ)

    def test_the_width_moves_with_a_pinned_reference_speed(self):
        env = self._half_space(5500.0)
        model = RAM(verbose=False, c0=1500.0)
        pinned = ram_domain.absorbing_layer_thickness(
            env, self.FREQ, knobs=model._knob_record(),
            speed_bounds=model._speed_bounds)
        assert pinned == pytest.approx(20.0 * 1500.0 / self.FREQ)

    def test_the_domain_the_span_and_the_ramp_use_the_one_width(self):
        """``zmax``, the mpiramS sediment span and the Collins ramp start
        are three readings of the same layer."""
        model = RAM(verbose=False, earth_curvature=False)
        env = self._half_space(2400.0)
        c0 = ram_domain.resolve_c0(env, knobs=model._knob_record(),
                                   speed_bounds=model._speed_bounds)
        width = model.absorber_width_wavelengths * c0 / self.FREQ
        zmax = ram_domain.compute_zmax(env, self.FREQ,
                                       max_range=5000.0, knobs=model._knob_record(),
                                       speed_bounds=model._speed_bounds)
        assert zmax - env.depth - ram_domain.absorber_span(
            env, self.FREQ, zmax, knobs=model._knob_record(),
            speed_bounds=model._speed_bounds) \
            == pytest.approx(width)
        (seg,) = ram_collins.collins_range_segments(
            env, 'ramgeo', zmax, self.FREQ, knobs=model._knob_record(),
            speed_bounds=model._speed_bounds)
        z_start = seg['bottom_attn'][-2][0]
        assert (zmax - env.depth) - z_start == pytest.approx(width)


class TestAbsorbingRampSpansTheAbsorbingWidthUnderAHalfSpace:
    """On the automatic grid every backend ramps its attenuation over exactly
    ``absorber_width_wavelengths`` wavelengths above the domain floor, and the ramp
    starts the seabed pad below the seafloor — the deeper of
    ``_SEABED_WAVELENGTHS_BEFORE_ABSORBER`` bottom wavelengths and
    ``leaky_field_depth`` — one rule for the four backends. A bare half-space is written as
    its two breakpoints, so nothing but the domain size decides where the
    ramp starts (``RAM.md`` p.2: the attenuation is "increased over the lower
    few wavelengths of the grid"). Without the ramp the grid floor is a
    pressure-release reflector under a flat-attenuation seabed
    (``ramgeo1.5.f:312-335`` updates ``u(2..nz+1)`` over a zeroed ``u``).
    """

    CASES = [(100.0, 3500.0), (1000.0, 300.0), (30.0, 2000.0)]
    C_BOTTOM = 1600.0

    @classmethod
    def _half_space(cls, depth, attn=0.5):
        return Environment(
            bathymetry=depth, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=cls.C_BOTTOM, density=1.8,
                                      attenuation=attn))

    @staticmethod
    def _absorbing_width(model, env, freq):
        return model.absorber_width_wavelengths * ram_domain.resolve_c0(
            env, knobs=model._knob_record(),
            speed_bounds=model._speed_bounds) / freq

    @pytest.mark.parametrize('kind', ['ramgeo', 'ramsurf', 'rams'])
    @pytest.mark.parametrize('depth, freq', CASES)
    # The ramp geometry is pinned in the geometric frame: under the default
    # ``earth_curvature`` the deck's depths are stretched by ``z/Re``.
    def test_the_collins_ramp_is_the_absorbing_width_wide(self, kind, depth,
                                                          freq):
        model = RAM(verbose=False, earth_curvature=False)
        env = self._half_space(depth)
        zmax = ram_domain.compute_zmax(env, freq, max_range=5000.0, knobs=model._knob_record(),
                                       speed_bounds=model._speed_bounds)
        (seg,) = ram_collins.collins_range_segments(
            env, kind, zmax, freq, knobs=model._knob_record(),
            speed_bounds=model._speed_bounds)
        attn = seg['bottom_attn']
        # ramgeo/ramsurf blocks are depth below the seafloor, rams absolute.
        z_floor = zmax - depth if kind in ('ramgeo', 'ramsurf') else zmax
        width = self._absorbing_width(model, env, freq)
        assert attn[-1][0] == pytest.approx(z_floor)
        assert attn[-1][1] == pytest.approx(model.absorber_attenuation)
        z_start, attn_start = attn[-2]
        assert z_start == pytest.approx(z_floor - width), (
            f"{kind}: ramp {z_floor - z_start:.2f} m wide, "
            f"absorbing width {width:.2f} m")
        assert attn_start == pytest.approx(0.5)
        # Everything above the ramp is the half-space itself.
        assert all(a == pytest.approx(0.5) for _, a in attn[:-1])

    @pytest.mark.parametrize('depth, freq', CASES)
    def test_the_ramp_starts_at_the_seabed_pad_below_the_seafloor(
            self, depth, freq):
        from uacpy.models.ram._domain import (
            _SEABED_WAVELENGTHS_BEFORE_ABSORBER,
        )
        model = RAM(verbose=False, earth_curvature=False)
        env = self._half_space(depth)
        zmax = ram_domain.compute_zmax(env, freq, max_range=5000.0, knobs=model._knob_record(),
                                       speed_bounds=model._speed_bounds)
        (seg,) = ram_collins.collins_range_segments(
            env, 'ramgeo', zmax, freq, knobs=model._knob_record(),
            speed_bounds=model._speed_bounds)
        pad = max(_SEABED_WAVELENGTHS_BEFORE_ABSORBER * self.C_BOTTOM / freq,
                  ram_domain.leaky_field_depth(env, freq, 5000.0,
                                               knobs=model._knob_record()))
        assert seg['bottom_attn'][-2][0] == pytest.approx(pad)

    @pytest.mark.parametrize('depth, freq', CASES)
    def test_the_mpirams_domain_and_ramp_follow_the_same_rule(self, depth,
                                                              freq, tmp_path):
        """mpiramS's ramp runs from control point ``nzs-1`` at
        ``seafloor + sedlayer`` to ``zmax`` (``ram.f90:334-342``), so its
        domain is sized by the same pad + absorbing width and ``sedlayer`` is
        that pad — no depth-fraction floor on either."""
        from uacpy.models.ram._domain import (
            _SEABED_WAVELENGTHS_BEFORE_ABSORBER,
        )
        model = RAM(verbose=False)
        env = self._half_space(depth)
        dz = 0.1
        pad = max(_SEABED_WAVELENGTHS_BEFORE_ABSORBER * self.C_BOTTOM / freq,
                  ram_domain.leaky_field_depth(env, freq, 5000.0,
                                               knobs=model._knob_record()))
        width = self._absorbing_width(model, env, freq)
        zmax = ram_mpirams.mpirams_zmax(env, freq, dz,
                                        max_range=5000.0, knobs=model._knob_record(),
                                        log=model._log,
                                        speed_bounds=model._speed_bounds)
        # Snapped onto the dz grid: within half a cell of the rule.
        assert zmax == pytest.approx(depth + max(dz, pad) + width,
                                     abs=0.5 * dz + 1e-3)
        span = ram_domain.absorber_span(env, freq, zmax,
                                        knobs=model._knob_record(),
                                        speed_bounds=model._speed_bounds)
        sedlayer = ram_mpirams.prepare_bottom_properties(
            env, tmp_path, span, zmax, dz=dz, knobs=model._knob_record(),
            log=model._log)[0]
        assert sedlayer == pytest.approx(span)
        assert span == pytest.approx(max(dz, pad), abs=0.5 * dz + 1e-3)

    def test_a_ramgeo_field_matches_a_grid_too_deep_for_its_floor_to_matter(
            self, tmp_path):
        """A lossless half-space returns everything the seabed does not
        absorb, so a grid floor the ramp does not shield shows as a level
        bias against a domain whose floor is far below the seabed. Levels
        are compared intensity-averaged over 100 m windows so interference
        fringes cannot be mistaken for a level error; ``dr``/``dz`` pinned
        so only ``zmax`` moves between the two runs. The deep grid moves the
        absorber itself, so coherent fringes still shift by ~0.8 dB rms even
        with the ramp in place; a deck whose ramp is 0.3 m wide sits at
        -0.5 dB mean / 1.7 dB max incoherent and 3.8 dB rms coherent."""
        env = self._half_space(100.0, attn=0.0)
        src = Source(depths=30.0, frequencies=3500.0)
        rcv = Receiver(depths=[50.0, 90.0],
                       ranges=np.arange(100.0, 1500.0, 10.0))

        def tl(zmax):
            kw = dict(backend='ramgeo', verbose=False, dr=1.0, dz=0.025,
                      work_dir=str(tmp_path / f'zmax_{zmax}'))
            if zmax is not None:
                kw['zmax'] = zmax
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return np.asarray(RAM(**kw).run(env, src, rcv).tl, dtype=float)

        def incoherent(tl_dB):
            intensity = 10.0 ** (-tl_dB / 10.0)
            windows = intensity.reshape(intensity.shape[0], -1, 10)
            return -10.0 * np.log10(windows.mean(axis=2))

        auto, deep = tl(None), tl(160.0)
        bias = incoherent(auto) - incoherent(deep)
        assert abs(float(bias.mean())) < 0.1, (
            f"incoherent level bias {bias.mean():+.2f} dB against 160 m")
        assert float(np.max(np.abs(bias))) < 0.3
        rms = float(np.sqrt(np.mean((auto - deep) ** 2)))
        assert rms < 1.5, f"coherent rms {rms:.2f} dB against zmax = 160 m"


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
            SeabedColumn.from_halfspace(hs))
        return Environment(bathymetry=[(0.0, 100.0), (10000.0, 300.0)],
                           ssp=1500.0, bottom=Bottom([col]))

    def test_layered_elastic_column_is_reanchored_along_the_slope(self):
        model = RAM()
        segs = ram_collins.collins_range_segments(
            self._env(layered=True), 'rams', zmax=700.0, freq=100.0,
            knobs=model._knob_record(), speed_bounds=model._speed_bounds)
        tops = [s['bottom_cs'][0][0] for s in segs]
        assert len(segs) > 10                       # was 1
        assert tops[0] == pytest.approx(100.0, abs=1.0)
        assert tops[-1] > 280.0                     # tracks to the deep end
        assert all(b >= a for a, b in zip(tops, tops[1:]))

    def test_half_space_column_gains_no_sections(self):
        # Both breakpoints of a half-space carry the same value, so anchoring
        # cannot matter and extra sections are pure cost.
        model = RAM()
        segs = ram_collins.collins_range_segments(
            self._env(layered=False), 'rams', zmax=700.0, freq=100.0,
            knobs=model._knob_record(), speed_bounds=model._speed_bounds)
        assert len(segs) == 1

    def test_seafloor_relative_backends_gain_no_sections(self):
        # ramgeo/ramsurf re-anchor in matrc already.
        for kind in ('ramgeo', 'ramsurf'):
            model = RAM()
            segs = ram_collins.collins_range_segments(
                self._env(layered=True), kind, zmax=700.0, freq=100.0,
                knobs=model._knob_record(), speed_bounds=model._speed_bounds)
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
                                  sound_speed=np.vstack([c, c]), ranges=r_ax),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                sound_speed=1700.0, density=1.8, attenuation=0.5)),
        )

    def _segments(self, n_cols):
        m = RAM(backend='ramgeo', dz=2.0, zmax=700.0, verbose=False)
        return m, ram_collins.collins_range_segments(self._env(n_cols),
                                                     'ramgeo',
                                            700.0, 50.0,
                                            knobs=m._knob_record(),
                                            speed_bounds=m._speed_bounds)

    @pytest.mark.parametrize('n_cols', [21, 51, 101, 201])
    def test_every_section_is_reachable_within_the_march(self, n_cols):
        m, segs = self._segments(n_cols)
        markers = sorted({float(s['range']) for s in segs})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dr = ram_collins.constrain_dr_to_sections(500.0, segs,
                                                      pinned=False, log=m._log)
        assert int(10000.0 / dr) >= len(markers) - 1, (
            f'{len(markers)} sections but only {int(10000.0/dr)} range steps '
            f'— the surplus would be silently dropped')

    def test_a_coarse_environment_leaves_dr_alone(self):
        # The discriminating counterpart: when the sections are already
        # further apart than dr there is nothing to constrain, and dr must
        # come back untouched rather than be clamped on every run.
        m, segs = self._segments(3)
        assert ram_collins.constrain_dr_to_sections(10.0, segs, pinned=False,
                                                    log=m._log) == 10.0

    def test_a_pinned_dr_warns_rather_than_being_changed_in_silence(self):
        # Single-config rule: uacpy may not quietly rewrite a knob the user
        # set. It still reduces dr — wrong numbers are worse than a warning.
        m, segs = self._segments(101)
        with pytest.warns(UserWarning, match='silently dropped'):
            dr = ram_collins.constrain_dr_to_sections(500.0, segs, pinned=True,
                                                      log=m._log)
        assert dr < 500.0


class TestSeafloorOutsideTheGrid:
    """``ram.pdf`` p.7 puts the grid bottom "well below the ocean bottom
    interface". Nothing enforces it — ``ramgeo1.5.f:133-135`` clamps
    ``iz=min(nz,iz)`` and ``rams0.5.f:135`` does not clamp at all — so a
    pinned ``zmax`` above the seafloor runs happily and returns numbers up to
    27.0 dB from an equivalent run with the seabed inside the grid."""

    ENV = Environment(
        bathymetry=220.0,
        ssp=SoundSpeedProfile(depths=[0.0, 220.0], sound_speed=[1500.0, 1500.0]),
        bottom=Bottom.from_halfspace(BoundaryProperties(
            sound_speed=1700.0, density=1.8, attenuation=0.5)))

    def _model(self, zmax):
        return RAM(backend='ramgeo', zmax=zmax, dz=2.0, dr=50.0, verbose=False)

    def test_pinned_zmax_above_the_seafloor_warns(self):
        with pytest.warns(UserWarning, match='outside the PE grid'):
            model = self._model(100.0)
            ram_domain.warn_if_seafloor_outside_grid(
                100.0, self.ENV, knobs=model._knob_record(),
                speed_bounds=model._speed_bounds)

    def test_a_grid_that_clears_the_seafloor_is_silent(self):
        # The discriminating counterpart — this must not warn on every run.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            model = self._model(700.0)
            ram_domain.warn_if_seafloor_outside_grid(
                700.0, self.ENV, knobs=model._knob_record(),
                speed_bounds=model._speed_bounds)

    def test_the_seafloor_index_follows_the_backend_that_will_run(self):
        """The seafloor INDEX formula is not shared across the family.

        rams0.5 writes ``iz = z/dz`` (``rams0.5.f:135``); the fluid codes and
        mpiramS write ``iz = 1 + z/dz`` and then clamp
        (``ramgeo1.5.f:133-135``, ``ramsurf1.5.f:118-120``,
        ``mpiramS/src/ram.f90:101``). On a 220 m seabed with dz=2 m and
        zmax=222 m, nz = int(222/2 - 0.5) = 110: rams' index is 110 and sits
        inside the grid, while the other three index 111 and sit one cell
        outside it. Reading rams' formula for all four called that case
        'inside' on three backends.
        """
        m = self._model(222.0)
        for kind in ('ramgeo', 'ramsurf', 'mpirams'):
            with pytest.warns(UserWarning, match='outside the PE grid'):
                ram_domain.warn_if_seafloor_outside_grid(222.0, self.ENV,
                                                         dz=2.0,
                                                 kind=kind, freq=50.0,
                                                 knobs=m._knob_record(),
                                                 speed_bounds=m._speed_bounds)
        # rams' own index is inside, so it gets the thin-margin diagnosis
        # instead — a different warning, not silence.
        with recorded_warnings() as caught:
            ram_domain.warn_if_seafloor_outside_grid(222.0, self.ENV, dz=2.0,
                                             kind='rams', freq=50.0,
                                             knobs=m._knob_record(),
                                             speed_bounds=m._speed_bounds)
        messages = [str(w.message) for w in caught]
        assert messages and not any('outside the PE grid' in t
                                    for t in messages)
        assert any('no room for the absorbing layer' in t for t in messages)

    def test_an_auto_zmax_is_never_flagged(self):
        # _compute_zmax clears the seafloor by construction, so only a pinned
        # value can reach the guard. The dz puts the 220 m seafloor a quarter
        # cell below a node, so no other notice fires either.
        m = RAM(backend='ramgeo', dz=220.0 / 110.25, dr=50.0, verbose=False)
        assert m.zmax is None
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            m.run(self.ENV, Source(depths=100.0, frequencies=50.0),
                  Receiver(depths=[50.0], ranges=np.array([1000.0, 2000.0])))


class TestSourceAgainstTheDepressedSurface:
    """``matrc`` zeroes every row down to ``izsrf``
    (``ramsurf1.5.f:281-290``) while ``selfs`` plants the source without
    consulting it (``:396-400``), so a source inside the depression yields
    an identically-zero field. ``ram.collins.mark_diverged_collins_samples``
    cannot catch it: ``|u| = 0`` is a large POSITIVE TL, not a negative or
    NaN one."""

    @staticmethod
    def _surface(depth, partial=False):
        return ([(0.0, 0.0), (5000.0, depth), (10000.0, depth)] if partial
                else [(0.0, depth), (10000.0, depth)])

    def _model(self):
        return RAM(backend='ramsurf', zmax=700.0, dz=2.0, dr=50.0,
                   verbose=False)

    def test_source_inside_the_depression_at_r0_is_refused(self):
        with pytest.raises(ConfigurationError, match='identically zero'):
            ram_collins.check_source_below_depressed_surface(
                self._surface(30.0), 10.0, 2.0)

    def test_a_keel_deeper_than_the_source_further_along_only_warns(self):
        # The dangerous variant: the field dies partway and reads as a
        # shadow zone. The near field is still meaningful, so warn.
        with pytest.warns(UserWarning, match='shadow zone'):
            ram_collins.check_source_below_depressed_surface(
                self._surface(30.0, partial=True), 10.0, 2.0)

    def test_the_message_states_the_observable_not_inf(self):
        # outpt adds eps=1e-20 before the log (ramsurf1.5.f:101), so a dead
        # field reports ~414-437 dB. Telling the user to look for inf sends
        # them after something that never appears.
        with pytest.raises(ConfigurationError,
                           match='is at or above the depressed surface') as exc:
            ram_collins.check_source_below_depressed_surface(
                self._surface(30.0), 10.0, 2.0)
        msg = str(exc.value)
        assert '414-437 dB' in msg, 'the message must name the observable'
        assert 'eps=1e-20' in msg, 'and why it is not inf'

    def test_a_source_below_every_depression_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            ram_collins.check_source_below_depressed_surface(
                self._surface(30.0), 100.0, 2.0)


class TestCollinsDivergenceIsMarkedNoDataNotOverwritten:
    """A diverged Collins march is reported, never repaired: the samples come
    back as NaN — the marker the wrapper already uses for receivers off the
    grid — instead of a level uacpy invented, which would read as a deep
    shadow zone. What separates divergence from a loud near field is the
    RANGE, not a fixed number of dB: TL is referenced to 1 m, so beyond that
    radius TL cannot go below 0, while inside it free-field spreading gives
    20*log10(r) — -2.5 dB at dr = 0.75 m and -44.7 dB where the lambda cap
    drives dr to 5.8 mm at 50 kHz. And it cannot be a NaN test, since
    ``rams0.5.f:265`` takes TL from ``alog10(cabs(ur))``: a blow-up under the
    float32 ceiling stays finite."""

    @staticmethod
    def _fluid_env():
        return make_pekeris(name='short')

    def test_a_finite_near_source_sample_keeps_its_value(self):
        src = Source(depths=50.0, frequencies=1000.0)
        rcv = Receiver(depths=[50.0], ranges=[0.75, 7.5, 30.0])
        with recorded_warnings() as caught:
            field = RAM(verbose=False, backend='ramgeo',
                        dr=0.75).compute_tl(self._fluid_env(), src, rcv)
        noise = [str(w.message) for w in caught
                 if 'divergence' in str(w.message)]
        assert not noise, noise
        dB = np.asarray(field.dB).ravel()
        assert np.isfinite(dB).all()
        # r = 0.75 m sits inside the 1 m reference radius, so its TL is
        # legitimately negative and is the march's own number.
        assert dB[0] < 0.0, f"near-source TL {dB[0]:.2f} dB"

    def test_a_nonfinite_grid_is_marked_no_data_and_warned(self):
        raw = {
            'tl': np.array([[60.0, np.inf, -50.0],
                            [70.0, 80.0, 90.0]]),
            'pcomplex': np.array([[1e-3, 1e6, 3e2],
                                  [1e-3, 1e-4, 1e-5]], dtype=complex),
            'ranges': np.array([10.0, 20.0, 30.0]),
            'frequency': 100.0,
        }
        with pytest.warns(UserWarning, match='returned as NaN') as caught:
            model = RAM(verbose=False)
            psi = ram_collins.mark_diverged_collins_samples(
                raw, self._fluid_env(), 'ramgeo', knobs=model._knob_record(),
                speed_bounds=model._speed_bounds)
        msgs = [str(w.message) for w in caught if 'NaN' in str(w.message)]
        # The non-finite sample and the -50 dB one at 30 m are both counted;
        # the advice must not name the measured-harmful remedy.
        assert any('2/6' in m for m in msgs), msgs
        assert all('smaller dr' not in m for m in msgs), msgs
        assert np.isnan(psi[0, 1]) and np.isnan(psi[0, 2])
        assert np.abs(psi[0, 0]) == pytest.approx(1e-3)

    def test_a_finite_but_diverged_grid_is_marked_no_data(self):
        """The regression a NaN gate introduces: rams0.5 writes TL through
        ``alog10``, so an elastic march that blows up without overflowing
        delivers finite samples hundreds of dB negative. Nothing in the grid
        is non-finite, and every such sample must still be caught."""
        raw = {
            'tl': np.array([[60.0, -2587.0, -900.0],
                            [70.0, 80.0, 90.0]]),
            'pcomplex': np.array([[1e-3, 1e30, 1e20],
                                  [1e-3, 1e-4, 1e-5]], dtype=complex),
            'ranges': np.array([100.0, 200.0, 300.0]),
            'frequency': 500.0,
        }
        with pytest.warns(UserWarning, match='returned as NaN') as caught:
            model = RAM(verbose=False)
            psi = ram_collins.mark_diverged_collins_samples(
                raw, self._fluid_env(), 'rams', knobs=model._knob_record(),
                speed_bounds=model._speed_bounds)
        assert any('2/6' in str(w.message) for w in caught)
        assert np.isnan(psi[0, 1]) and np.isnan(psi[0, 2])

    def test_the_allowed_level_follows_the_range(self):
        """Both sides of the rule at once: -30 dB is the march diverging at
        30 m, where TL cannot go below 0, and an ordinary value at 5.8 mm —
        the lambda-capped first step at 50 kHz, where free-field spreading
        alone gives -44.7 dB. A fixed dB threshold cannot separate these."""
        raw = {
            'tl': np.array([[-30.0, -44.0]]),
            'pcomplex': np.array([[31.6, 158.0]], dtype=complex),
            'ranges': np.array([30.0, 0.0058]),
            'frequency': 50000.0,
        }
        with pytest.warns(UserWarning, match='returned as NaN') as caught:
            model = RAM(verbose=False)
            psi = ram_collins.mark_diverged_collins_samples(
                raw, self._fluid_env(), 'ramgeo', knobs=model._knob_record(),
                speed_bounds=model._speed_bounds)
        assert any('1/2' in str(w.message) for w in caught)
        assert np.isnan(psi[0, 0])
        assert np.abs(psi[0, 1]) == pytest.approx(158.0)

    def test_a_few_dB_of_near_field_gain_is_not_divergence(self):
        """The deepest legitimate negative at dr = 0.75 m is -2.49 dB, and it
        is the engine's own number: no warning, magnitude uncapped."""
        raw = {
            'tl': np.array([[-2.49, 40.0]]),
            'pcomplex': np.array([[1.33, 1e-2]], dtype=complex),
            'ranges': np.array([0.75, 100.0]),
            'frequency': 1000.0,
        }
        with recorded_warnings() as caught:
            model = RAM(verbose=False)
            psi = ram_collins.mark_diverged_collins_samples(
                raw, self._fluid_env(), 'ramgeo', knobs=model._knob_record(),
                speed_bounds=model._speed_bounds)
        assert not [w for w in caught if 'NaN' in str(w.message)]
        assert np.abs(psi)[0, 0] == pytest.approx(1.33)


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
        ssp=SoundSpeedProfile(depths=[0.0, 220.0], sound_speed=[1500.0, 1500.0]),
        bottom=Bottom.from_halfspace(BoundaryProperties(
            sound_speed=1700.0, density=1.8, attenuation=0.5)))

    @pytest.mark.parametrize('backend', ['ramgeo', 'ramsurf', 'rams',
                                         'mpirams'])
    def test_a_pinned_zmax_below_the_seafloor_warns_on_every_backend(
            self, backend):
        m = RAM(backend=backend, zmax=200.0, dz=2.0, dr=50.0, verbose=False)
        with pytest.warns(UserWarning, match='outside the PE grid'):
            ram_domain.compute_zmax(self.ENV, 50.0, max_range=5000.0, knobs=m._knob_record(),
                                    speed_bounds=m._speed_bounds)

    @pytest.mark.parametrize('backend', ['ramgeo', 'mpirams'])
    def test_a_grid_that_clears_the_seafloor_stays_silent(self, backend):
        m = RAM(backend=backend, zmax=1500.0, dz=2.0, dr=50.0, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            ram_domain.compute_zmax(self.ENV, 50.0, max_range=5000.0, knobs=m._knob_record(),
                                    speed_bounds=m._speed_bounds)

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
        surface = [(0.0, zsrf), (10000.0, zsrf)]
        if alive:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ram_collins.check_source_below_depressed_surface(surface, zs,
                                                                 dz)
        else:
            with pytest.raises(ConfigurationError,
                               match='is at or above the depressed surface'):
                ram_collins.check_source_below_depressed_surface(surface, zs,
                                                                 dz)


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
        ssp=SoundSpeedProfile(depths=[0.0, 200.0], sound_speed=[1500.0, 1500.0]),
        bottom=Bottom.from_halfspace(BoundaryProperties(
            sound_speed=1800.0, density=2.0, attenuation=0.1,
            shear_speed=800.0, shear_attenuation=0.2)))

    def _model(self, backend, zmax):
        return RAM(backend=backend, zmax=zmax, dz=1.0, dr=25.0, verbose=False)

    def test_rams_refuses_a_zmax_that_pushes_iz_past_nz(self):
        # 0.3 m below the seabed: zmax > depth, so a depth-only test passes.
        with pytest.raises(ConfigurationError, match='iz > nz|does not.*clamp'):
            model = self._model('rams', 200.3)
            ram_collins.resolve_collins_grid(
                self.ENV, 100.0, 'rams', 6000.0, None, None, None,
                knobs=model._knob_record(), log=model._log,
                speed_bounds=model._speed_bounds)

    def test_half_a_cell_more_is_accepted(self):
        # The discriminating counterpart: iz == nz is in-domain and the run
        # returns finite numbers. The guard must not creep upward.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = self._model('rams', 200.6)
            ram_collins.resolve_collins_grid(
                self.ENV, 100.0, 'rams', 6000.0, None, None, None,
                knobs=model._knob_record(), log=model._log,
                speed_bounds=model._speed_bounds)

    @pytest.mark.parametrize('backend', ['ramgeo', 'ramsurf'])
    def test_the_clamped_backends_only_warn(self, backend):
        # They apply min(nz,iz) and return usable numbers, so refusing them
        # would reject a run that works.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = self._model(backend, 200.3)
            ram_collins.resolve_collins_grid(
                self.ENV, 100.0, backend, 6000.0, None, None, None,
                knobs=model._knob_record(), log=model._log,
                speed_bounds=model._speed_bounds)


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

    The threshold moves with the auto ``dz``, so these cases are stated against
    the resolved grid rather than a remembered number.
    ``ram.grid.align_dz_with_seafloor`` takes the coarsest ``dz`` that divides
    the shelf depth and still clears the shear-wavelength cap — 5.714 m here
    (c_s = 800 m/s, 10 Hz). So a shelf *deeper* than that cap gets split into
    at least two cells and is in-domain (6 m → dz 3.0, iz 2; 8 m → dz 4.0, iz
    2), while a shelf *at or under* the cap cannot be split at all: the
    coarsest divisor is the depth itself, the whole water column becomes one
    cell, and ``iz`` falls to 1 (5 m → dz 5.0; 3 m → dz 3.0). The refusal
    boundary is therefore the shear cap, and both sides of it are pinned below.
    """

    @staticmethod
    def _env(shelf):
        return Environment(
            bathymetry=Bathymetry(ranges=[0.0, 8000.0, 20000.0],
                                  depths=[200.0, shelf, shelf]),
            ssp=SoundSpeedProfile(depths=[0.0, 200.0], sound_speed=[1500.0, 1500.0]),
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
        assert np.isfinite(f.dB).all()

    def test_the_clamped_backends_are_unaffected(self):
        # ramgeo applies max(2,iz) (ramgeo1.5.f:134), so the same track is
        # fine there and refusing it would be wrong.
        env = Environment(
            bathymetry=Bathymetry(ranges=[0.0, 8000.0, 20000.0],
                                  depths=[200.0, 8.0, 8.0]),
            ssp=SoundSpeedProfile(depths=[0.0, 200.0], sound_speed=[1500.0, 1500.0]),
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
        ssp=SoundSpeedProfile(depths=[0.0, 220.0], sound_speed=[1500.0, 1500.0]),
        bottom=Bottom.from_halfspace(BoundaryProperties(
            sound_speed=1700.0, density=1.8, attenuation=0.5)))

    def _warns(self, zmax):
        m = RAM(backend='ramgeo', zmax=zmax, dz=2.0, dr=50.0, verbose=False)
        with recorded_warnings() as caught:
            ram_domain.warn_if_seafloor_outside_grid(zmax, self.ENV, dz=2.0,
                                             kind='ramgeo', freq=50.0,
                                             knobs=m._knob_record(),
                                             speed_bounds=m._speed_bounds)
        return any('grid' in str(w.message) or 'absorbing' in str(w.message)
                   for w in caught)

    @pytest.mark.parametrize('zmax,measured_dB', [(300.0, 3.1), (222.0, 16.0),
                                                  (220.0, 27.0)])
    def test_a_thin_sub_bottom_margin_warns(self, zmax, measured_dB):
        assert self._warns(zmax), f'{measured_dB} dB error passed in silence'

    @pytest.mark.parametrize('zmax,measured_dB', [(700.0, 0.32), (1500.0, 0.0)])
    def test_an_ample_margin_stays_silent(self, zmax, measured_dB):
        # The discriminating half. Comparing against uacpy's own 20-lambda
        # auto pad would fire here, at 0.32 dB — a warning users would learn
        # to ignore.
        assert not self._warns(zmax), \
            f'{measured_dB} dB error warned — the threshold is too eager'


class TestSourceRowIsActuallySolved:
    """Every RAM binary plants the source with ``si=1.0+zs/dz`` / ``is=ifix(si)``
    and splits it across ``u(is)``, ``u(is+1)`` (``ramgeo1.5.f:389-393``,
    ``ramsurf1.5.f:396-400``, ``rams0.5.f:357-361``,
    ``mpiramS/src/ram.f90:110-114``), and every solver sweeps from row 2
    (``ramgeo1.5.f:319``, ``rams0.5.f:838``, ``solvetri.f90:46``).

    So ``zs < dz`` freezes part of the source in ``u(1)`` for the whole march,
    where it acts as a permanent Dirichlet source on the pressure-release
    surface: the field gets *louder* as the source approaches the surface,
    which is backwards. Measured against **Kraken** as an independent arbiter
    — never backend-vs-backend — at ~46 dB mean and 72 dB peak on uacpy's own
    default grid, silent on every backend.

    This is the flat-surface case of the row-1 kill that
    ``ram.collins.check_source_below_depressed_surface`` catches for a ramsurf
    keel; it needs no altimetry and is not ramsurf-specific.
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
            ssp=SoundSpeedProfile(depths=[0.0, 100.0], sound_speed=[1500.0, 1500.0]),
            bottom=Bottom.from_halfspace(bp), **kw)

    RCV = Receiver(depths=[10.0, 50.0, 90.0],
                   ranges=np.linspace(500.0, 5000.0, 10))

    @pytest.mark.parametrize('backend', ['ramgeo', 'ramsurf', 'rams',
                                         'mpirams'])
    def test_a_source_inside_the_first_cell_is_refused(self, backend):
        # A pinned dz is the caller's; the automatic grid comes down to the
        # source instead (``ram.grid.compute_grid_lytaev(zs=...)``).
        env = self._env(elastic=(backend == 'rams'),
                        altimetry=(backend == 'ramsurf'))
        with pytest.raises(ConfigurationError, match='shallower than one'):
            RAM(backend=backend, verbose=False, dz=0.55).run(
                env, Source(depths=0.5, frequencies=100.0), self.RCV)

    @pytest.mark.parametrize('backend', ['ramgeo', 'mpirams'])
    def test_a_source_below_the_first_cell_runs(self, backend):
        # The discriminating counterpart.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = RAM(backend=backend, verbose=False).run(
                self._env(), Source(depths=25.0, frequencies=100.0), self.RCV)
        assert np.isfinite(f.dB).any()

    def test_the_guard_is_keyed_to_dz_not_to_a_fixed_depth(self):
        # A 0.5 m source is fine once dz is small enough to resolve it — the
        # bound is the mechanism (is >= 2), not an absolute shallow-source
        # rule.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = RAM(backend='ramgeo', dz=0.1, dr=50.0, verbose=False).run(
                self._env(), Source(depths=0.5, frequencies=100.0), self.RCV)
        assert np.isfinite(f.dB).any()


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
            ssp=SoundSpeedProfile(depths=[0.0, 100.0], sound_speed=[1500.0, 1500.0]),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                sound_speed=1700.0, density=1.8, attenuation=0.5)))
        m = self._model()
        segs = ram_collins.collins_range_segments(env, 'ramgeo', 200.0, 100.0,
                                                  knobs=m._knob_record(),
                                                  speed_bounds=m._speed_bounds)
        bathy = [float(x) for x, _ in env.bathymetry.to_pairs().tolist()]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = ram_collins.constrain_dr_to_sections(250.0, segs,
                                                       pinned=False,
                                              bathymetry_ranges=bathy,
                                              log=m._log)
        assert out < 50.0
        assert out == pytest.approx(50.0 * (1 - 1e-4))

    def test_coarse_bathymetry_leaves_dr_alone(self):
        env = Environment(
            bathymetry=self._bathy(11),           # 500 m spacing
            ssp=SoundSpeedProfile(depths=[0.0, 100.0], sound_speed=[1500.0, 1500.0]),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                sound_speed=1700.0, density=1.8, attenuation=0.5)))
        m = self._model()
        segs = ram_collins.collins_range_segments(env, 'ramgeo', 200.0, 100.0,
                                                  knobs=m._knob_record(),
                                                  speed_bounds=m._speed_bounds)
        bathy = [float(x) for x, _ in env.bathymetry.to_pairs().tolist()]
        assert ram_collins.constrain_dr_to_sections(
            250.0, segs, pinned=False, bathymetry_ranges=bathy,
            log=m._log) == 250.0

    def test_a_constructor_pinned_dr_warns_before_being_changed(self):
        # Single-config rule: dr set on __init__ is as user-set as one passed
        # to run(), and must not be rewritten in silence.
        env = Environment(
            bathymetry=self._bathy(101),
            ssp=SoundSpeedProfile(depths=[0.0, 100.0], sound_speed=[1500.0, 1500.0]),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                sound_speed=1700.0, density=1.8, attenuation=0.5)))
        m = RAM(backend='ramgeo', dr=250.0, dz=0.5, verbose=False)
        segs = ram_collins.collins_range_segments(env, 'ramgeo', 200.0, 100.0,
                                                  knobs=m._knob_record(),
                                                  speed_bounds=m._speed_bounds)
        bathy = [float(x) for x, _ in env.bathymetry.to_pairs().tolist()]
        with pytest.warns(UserWarning, match='silently dropped'):
            ram_collins.constrain_dr_to_sections(250.0, segs, pinned=True,
                                        bathymetry_ranges=bathy, log=m._log)


class TestSedimentOffsetsFollowTheSeafloorBetweenBottomBreaks:
    """mpiramS rebuilds the seabed speed as ``csg = cwg + cs`` against the
    LOCAL water column (``ram.f90:345-346``) and marches with the nearest
    written profile (``:316-320``), so ``cs`` has to be re-referenced
    wherever the water speed at the seafloor moves — the bathymetry samples
    and SSP breaks of
    :func:`~uacpy.models.ram.mpirams.seafloor_speed_ranges` — on a
    range-dependent bottom exactly as on a range-independent one. A profile
    written only at the bottom's own breaks leaves every range between them
    referenced to a seafloor that is not its own, drifting by
    ``|dc/dz|·Δseafloor``."""

    @staticmethod
    def _env():
        # 100 → 200 m slope under a 0.1 s⁻¹ downward-refracting gradient, on
        # a half-space that steps from 1600 to 1700 m/s at 3 km.
        # rho_w = 1 keeps the recorded density rows about the seafloor
        # reference, not the water density.
        return Environment(
            name='rd-halfspace-gradient', water_density=1.0,
            bathymetry=Bathymetry(ranges=[0.0, 5000.0], depths=[100.0, 200.0]),
            ssp=SoundSpeedProfile.from_pairs(
                np.array([[0.0, 1520.0], [200.0, 1500.0]])),
            bottom=Bottom.from_halfspaces(
                np.array([0.0, 3000.0]),
                sound_speed=np.array([1600.0, 1700.0]),
                density=np.array([1.6, 1.8]),
                attenuation=np.array([0.5, 0.3])))

    def _profiles(self, monkeypatch, tmp_path):
        model = RAM(backend='mpirams', verbose=False, dz=0.5)
        env = self._env()
        written = {}

        def capture(work_dir, ranges, cs, rho, attn):
            written.update(ranges=np.asarray(ranges, float), cs=cs,
                           rho=rho, attn=attn)
            return 'sediment.sed'

        monkeypatch.setattr(
            'uacpy.models.ram.mpirams.write_sediment_profiles', capture)
        zmax = ram_mpirams.mpirams_zmax(env, 100.0, 0.5,
                                        max_range=5000.0, knobs=model._knob_record(),
                                        log=model._log,
                                        speed_bounds=model._speed_bounds)
        span = ram_domain.absorber_span(env, 100.0, zmax,
                                        knobs=model._knob_record(),
                                        speed_bounds=model._speed_bounds)
        sedlayer, nzs, *_rest, isedrd, _ = (
            ram_mpirams.prepare_bottom_properties(
                env, tmp_path, span, zmax, dz=0.5,
                knobs=model._knob_record(), log=model._log))
        assert isedrd == 1
        return model, env, written, sedlayer, nzs, zmax

    def test_profiles_are_written_where_the_seafloor_speed_moves(
            self, monkeypatch, tmp_path):
        model, env, written, *_ = self._profiles(monkeypatch, tmp_path)
        expected = set(ram_mpirams.seafloor_speed_ranges(env)) | {0.0, 3000.0}
        assert expected <= set(written['ranges'].tolist())

    def test_a_mid_break_profile_is_referenced_to_its_own_seafloor(
            self, monkeypatch, tmp_path):
        model, env, written, sedlayer, nzs, zmax = self._profiles(monkeypatch,
                                                                  tmp_path)
        ranges = written['ranges']
        # A written range strictly inside the first bottom column, away
        # from both of its breaks.
        inside = [r for r in ranges if 500.0 < r < 2500.0]
        assert inside, ranges
        r = inside[len(inside) // 2]
        i = int(np.where(ranges == r)[0][0])
        seafloor = float(np.asarray(env.bathymetry.eval(range=r)).flat[0])
        z_ctrl = ram_mpirams.control_point_depths(seafloor, sedlayer, nzs,
                                                  zmax)
        rebuilt = ram_domain.ssp_column(env, r, z_ctrl) + written['cs'][:, i]
        assert rebuilt[1:] == pytest.approx(1600.0)
        assert written['rho'][1:, i] == pytest.approx(1.6)

    def test_the_column_switch_stays_at_the_bottoms_own_midpoint(
            self, monkeypatch, tmp_path):
        # ``Bottom.at`` is nearest, so the 1600 → 1700 m/s switch belongs
        # midway between the 0 and 3 km breaks; mpiramS's nearest-profile
        # rule over profiles ≤ span/127 apart moves it by at most half that.
        model, env, written, *_ = self._profiles(monkeypatch, tmp_path)
        ranges, rho = written['ranges'], written['rho']
        k = int(np.argmax(rho[-2, :] > 1.7))
        assert k > 0
        switch = 0.5 * (ranges[k - 1] + ranges[k])
        assert switch == pytest.approx(1500.0, abs=5000.0 / 254.0)


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
        with pytest.warns(UserWarning, match=r"RAM: 1 receiver range\(s\) at r = 0") as record:
            field = RAM(verbose=False).run(env, src, rcv,
                                           run_mode=RunMode.COHERENT_TL)
        r0_warnings = [w for w in record
                       if 'r = 0' in str(w.message)]
        assert len(r0_warnings) == 1, (
            f"expected the shared source-axis warning once, got "
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
            tf = RAM(verbose=False, q_factor=4.0, record_duration=0.4, angle_max=45.0).run(
                env, src, rcv, run_mode=RunMode.BROADBAND)
            trace = tf.to_time_trace(depth=50.0, range=5000.0)
        assert tf.run_settings.engine.c0 > 1600.0
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


class TestCMinIsOneQuantityOnEveryPath:
    """``run_settings.waveguide.c_min`` is the slowest compressional speed
    ANYWHERE in the environment — water column plus seabed — on every
    backend and every run mode, the mirror of ``c_max``, and the result
    carries it there alone. A mud half-space slower than the water separates
    that from the water-column minimum, which is the other quantity it could
    plausibly be: 1450 m/s against 1500.

    The environment minimum is the one the waveguide records, because it is
    what the automatic grid chooser floors ``dz`` on (``models/ram/_model.py`` binds
    ``c_min_all`` from ``_speed_bounds``, not from
    ``ram._domain.water_speed_bounds``). mpiramS's own header ``cmin`` is the
    *water* minimum (``peramx.f90:295``) and is ``speeds.water_min`` instead.
    """

    WATER_SPEED = 1500.0
    SEABED_SPEED = 1450.0

    def _env(self):
        return Environment(
            bathymetry=100.0, ssp=self.WATER_SPEED,
            bottom=BoundaryProperties(
                acoustic_type='half-space', sound_speed=self.SEABED_SPEED,
                density=1.5, attenuation=0.5))

    @staticmethod
    def _geometry():
        return (Source(depths=25.0, frequencies=100.0),
                Receiver(depths=np.array([50.0]),
                         ranges=np.array([1000.0, 2000.0])))

    @pytest.mark.parametrize('backend', ['mpirams', 'ramgeo'])
    def test_narrowband_stamps_the_environment_minimum(self, backend):
        env = self._env()
        src, rcv = self._geometry()
        field = RAM(verbose=False, backend=backend).run(env, src, rcv)
        c_min = field.run_settings.waveguide.c_min
        assert c_min == pytest.approx(self.SEABED_SPEED), (
            f"{backend} narrowband c_min={c_min} is not the environment "
            f"minimum {self.SEABED_SPEED}")
        assert 'c_min' not in field.metadata

    def test_broadband_mpirams_stamps_the_environment_minimum(self):
        env = self._env()
        src, rcv = self._geometry()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tf = RAM(verbose=False, backend='mpirams', q_factor=4.0, record_duration=0.4).run(
                env, src, rcv, run_mode=RunMode.BROADBAND)
        assert tf.run_settings.waveguide.c_min == pytest.approx(
            self.SEABED_SPEED)

    def test_broadband_mpirams_keeps_the_binary_water_minimum_apart(self):
        """Both quantities are on the result, under different keys, and they
        are different numbers on this environment — which is what makes one
        key carrying both a defect rather than a naming preference."""
        env = self._env()
        src, rcv = self._geometry()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tf = RAM(verbose=False, backend='mpirams', q_factor=4.0, record_duration=0.4).run(
                env, src, rcv, run_mode=RunMode.BROADBAND)
        assert tf.speeds.water_min == pytest.approx(self.WATER_SPEED)
        assert tf.run_settings.waveguide.c_min == pytest.approx(
            self.SEABED_SPEED)


def _elastic_halfspace():
    return BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0, shear_speed=400.0,
        density=1.8, attenuation=0.5, shear_attenuation=0.5)


class TestConstructorGuards:
    """Pure-Python constructor contracts; no binary is launched."""

    @pytest.mark.parametrize('bad', [1, 11, 0, -3])
    def test_n_pade_outside_2_to_10_raises(self, bad):
        with pytest.raises(ConfigurationError, match=r'n_pade'):
            RAM(n_pade=bad, verbose=False)

    def test_n_pade_must_be_an_integer(self):
        with pytest.raises(ConfigurationError, match=r'n_pade'):
            RAM(n_pade=6.0, verbose=False)

    def test_rams_rotation_angle_accepts_a_callable(self):
        m = RAM(rams_rotation_angle=lambda f: 20.0 + f / 100.0, verbose=False)
        assert ram_stability.theta_for_freq(
            200.0, knobs=m._knob_record()) == pytest.approx(22.0)
        assert ram_stability.theta_for_freq(
            500.0, knobs=m._knob_record()) == pytest.approx(25.0)

    def test_rams_rotation_angle_float_out_of_range_raises(self):
        with pytest.raises(ConfigurationError, match=r'rams_rotation_angle'):
            RAM(rams_rotation_angle=95.0, verbose=False)

    def test_low_absorber_attenuation_warns(self):
        """Below 1 dB/λ the artificial absorber lets domain-bottom
        reflections back into the field (the RAM wrapper's Collins-readme note)."""
        with pytest.warns(UserWarning, match='absorber_attenuation'):
            RAM(absorber_attenuation=0.5, verbose=False)

    def test_ordinary_absorber_attenuation_is_silent(self):
        with recorded_warnings() as caught:
            RAM(absorber_attenuation=5.0, verbose=False)
        assert not [w for w in caught
                    if 'absorber_attenuation' in str(w.message)]

    @pytest.mark.parametrize('name, bad', [
        ('dr', '5'), ('q_factor', 'x'), ('dr', True), ('depth_decimation', True)])
    def test_a_knob_that_is_not_a_number_is_refused_by_name(self, name,
                                                            bad):
        """A string or a ``bool`` in a numeric knob is refused with the
        knob's name (the shared validators of ``models._knobs``), not taken
        as a number or left to fail inside numpy."""
        with pytest.raises(ConfigurationError, match=name):
            RAM(verbose=False, **{name: bad})

    def test_a_negative_stability_count_states_the_bound(self):
        with pytest.raises(ConfigurationError,
                           match=r'n_stability must be an integer >= 0'):
            RAM(verbose=False, n_stability=-1)

    @pytest.mark.parametrize('name,bad,good', [
        ('dr', -1.0, 5.0), ('n_pade', 11, 10), ('rams_rotation', 1, False),
        ('rams_dr_factor', 0.5, 1.0), ('backend', 'pe', 'ramgeo'),
        ('n_sediment_points', 3, 4)])
    def test_a_knob_reassigned_after_construction_is_checked_by_the_run(
            self, name, bad, good):
        """Stage 2 re-runs the constructor's checks (``_check_knobs``):
        ``validate_inputs`` and ``run`` refuse the value the constructor
        refuses, with its message, before any deck is written, and accept
        the legal value at the bound."""
        env = uacpy.Environment(name='k', bathymetry=100.0, ssp=1500.0)
        src = uacpy.Source(depths=50.0, frequencies=100.0)
        rcv = uacpy.Receiver(depths=[20.0, 50.0], ranges=[500.0, 1000.0])
        model = RAM(verbose=False)
        setattr(model, name, bad)
        with pytest.raises(ConfigurationError, match=name):
            model.validate_inputs(env, src, rcv)
        with pytest.raises(ConfigurationError, match=name):
            model.run(env, src, rcv)
        setattr(model, name, good)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.validate_inputs(env, src, rcv)


def _fluid_and_elastic_envs():
    """A 100-m guide over a 1600 m/s fluid half-space, and the same guide over
        ``_elastic_halfspace()``."""
    fluid = Environment(
        bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1600.0, density=1.5,
                                  attenuation=0.5))
    elastic = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=_elastic_halfspace())
    return fluid, elastic


class TestMarkDivergedCollinsSamples:
    """``ram.collins.mark_diverged_collins_samples`` reports one Collins run
    without repairing it: NaN/inf, and any TL below what its range allows,
    come back as NaN with a counting UserWarning; every other sample is the
    engine's own number, phase and magnitude untouched — including the exact
    zero at the z = 0 pressure-release node, which the shared dB conversion
    floors."""

    FLOOR = 10.0 ** (-200.0 / 20.0)

    @staticmethod
    def _raw():
        pcomplex = np.array([
            0.3 * np.exp(0.7j),      # valid sample
            2e30 * np.exp(0.3j),     # TL = -600 dB: divergence
            np.nan + 1j * np.nan,    # NaN
            5.0 + 0.0j,              # TL = +inf marker below
            1.06 * np.exp(-0.2j),    # TL = -0.5 dB, read at 0.75 m
            0.0 + 0.0j,              # surface node
        ])
        tl = np.array([10.46, -600.0, np.nan, np.inf, -0.5, 200.0])
        # The -0.5 dB sample sits inside the 1 m reference radius, where
        # free-field spreading allows -2.5 dB; the rest are past it, where
        # nothing below 0 dB is physical.
        ranges = np.array([10.0, 100.0, 200.0, 300.0, 0.75, 500.0])
        return {'tl': tl, 'pcomplex': pcomplex, 'ranges': ranges,
                'frequency': 100.0}

    def test_diverged_samples_come_back_as_no_data(self):
        fluid, _ = _fluid_and_elastic_envs()
        m = RAM(verbose=False)
        with pytest.warns(UserWarning, match=r'3/6 TL samples') as rec:
            out = ram_collins.mark_diverged_collins_samples(
                self._raw(), fluid, 'ramgeo', knobs=m._knob_record(),
                speed_bounds=m._speed_bounds)
        assert 'returned as NaN' in str(rec[0].message)
        assert np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3])

    def test_valid_samples_pass_through_unchanged(self):
        fluid, _ = _fluid_and_elastic_envs()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = RAM(verbose=False)
            out = ram_collins.mark_diverged_collins_samples(
                self._raw(), fluid, 'ramgeo', knobs=model._knob_record(),
                speed_bounds=model._speed_bounds)
        assert out[0] == pytest.approx(0.3 * np.exp(0.7j))

    def test_near_field_gain_and_surface_node_are_not_divergence(self):
        fluid, _ = _fluid_and_elastic_envs()
        raw = self._raw()
        # Keep only the three uncounted kinds: valid, near-field gain, zero.
        for key in ('tl', 'pcomplex', 'ranges'):
            raw[key] = raw[key][[0, 4, 5]]
        with recorded_warnings() as caught:
            model = RAM(verbose=False)
            out = ram_collins.mark_diverged_collins_samples(
                raw, fluid, 'ramgeo', knobs=model._knob_record(),
                speed_bounds=model._speed_bounds)
        assert not [w for w in caught if 'TL samples' in str(w.message)]
        # |p/p0| above 1 read inside the 1 m reference radius is the field,
        # and comes back as computed.
        assert out[1] == pytest.approx(1.06 * np.exp(-0.2j))
        # The z = 0 pressure-release node keeps its literal zero: the shared
        # dB conversion floors it to the one no-energy level uacpy reports,
        # rather than this wrapper writing a second one.
        assert out[2] == 0.0 + 0.0j

    def test_rams_divergence_names_the_alternatives_for_shear_above_c0(self):
        """The rams note names OAST / Scooter only when a shear speed
        exceeds the PE reference speed, where the rotated march loses
        accuracy fastest; an ordinary slow-shear seabed that lost a sample
        gets the grid advice alone."""
        _, elastic = _fluid_and_elastic_envs()
        model = RAM(verbose=False)
        assert ram_domain.max_shear_speed(
            elastic) < ram_domain.resolve_c0(elastic,
                                             knobs=model._knob_record(),
                                             speed_bounds=model._speed_bounds)
        with recorded_warnings() as caught:
            ram_collins.mark_diverged_collins_samples(
                self._raw(), elastic, 'rams', knobs=model._knob_record(),
                speed_bounds=model._speed_bounds)
        assert not [w for w in caught if 'OAST' in str(w.message)]
        fast = Environment(bathymetry=100.0, ssp=1500.0,
                           bottom=BoundaryProperties(
                               acoustic_type='half-space', sound_speed=5500.0,
                               density=2.6, attenuation=0.1,
                               shear_speed=3000.0, shear_attenuation=0.2))
        assert 3000.0 > ram_domain.resolve_c0(fast, knobs=model._knob_record(),
                                              speed_bounds=model._speed_bounds)
        with pytest.warns(UserWarning, match='OAST / Scooter'):
            ram_collins.mark_diverged_collins_samples(
                self._raw(), fast, 'rams', knobs=model._knob_record(),
                speed_bounds=model._speed_bounds)

    def test_surviving_samples_are_the_engines_own_bits(self):
        """Every valid sample is returned as read — equal to the input
        bit for bit, not rebuilt from its magnitude and phase (which lands
        one ULP off on a large share of samples)."""
        fluid, _ = _fluid_and_elastic_envs()
        rng = np.random.default_rng(7)
        n = 20000
        psi = (10.0 ** rng.uniform(-6.0, 2.0, n)
               * np.exp(1j * rng.uniform(-np.pi, np.pi, n)))
        raw = {'tl': -20.0 * np.log10(np.abs(psi)) + 60.0,
               'pcomplex': psi, 'ranges': np.full(n, 1000.0),
               'frequency': 100.0}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = RAM(verbose=False)
            out = ram_collins.mark_diverged_collins_samples(
                raw, fluid, 'ramgeo', knobs=model._knob_record(),
                speed_bounds=model._speed_bounds)
        assert np.array_equal(out, psi)


def _stub_grid(monkeypatch, model, dr, dz):
    """Make ``optimize_grid_relaxing`` return ``dr``/``dz`` unchanged; returns ``model``."""
    def fake(**kw):
        return {'dr': dr, 'dz': dz}, kw['eps0'], kw['theta0']
    monkeypatch.setattr(
        'uacpy.models.ram.grid.optimize_grid_relaxing', fake)
    return model


class TestGridResolverConstraints:
    """The Lytaev optimiser minimises its error model alone; ram.md §5 lists
    the constraints applied to its output afterwards. Feeding
    ``ram.grid.compute_grid_lytaev`` a known optimiser result via a stub
    isolates exactly those constraints."""

    def test_rams_dr_factor_divides_the_optimum(self, monkeypatch):
        # dr_opt = 10 → safety 10/5 = 2 m, tighter than the λ cap
        # c_min/(5f) = 1500/500 = 3 m.
        _, elastic = _fluid_and_elastic_envs()
        m = _stub_grid(monkeypatch, RAM(backend='rams', verbose=False),
                       dr=10.0, dz=1.0)
        dr, _ = ram_grid.compute_grid_lytaev(elastic, 100.0, max_range=5000.0,
                                       kind='rams', knobs=m._knob_record(),
                                       log=m._log,
                                       speed_bounds=m._speed_bounds)
        assert dr == pytest.approx(2.0)

    def test_rams_wavelength_cap_binds_when_tighter(self, monkeypatch):
        # With the safety factor disabled the λ cap c_min/(5f) = 3 m rules.
        _, elastic = _fluid_and_elastic_envs()
        m = _stub_grid(monkeypatch, RAM(backend='rams',
                                        rams_dr_factor=1.0,
                           verbose=False), dr=10.0, dz=1.0)
        dr, _ = ram_grid.compute_grid_lytaev(elastic, 100.0, max_range=5000.0,
                                       kind='rams', knobs=m._knob_record(),
                                       log=m._log,
                                       speed_bounds=m._speed_bounds)
        assert dr == pytest.approx(1500.0 / (5.0 * 100.0))

    def test_the_stability_rule_caps_the_step_where_the_wavelength_cap_does_not(
            self, monkeypatch):
        """At 5 kHz over 100 m of sand the Crank-Nicolson growth outruns the
        seabed's leak below the λ/5 cap (0.060 m), so the stability rule
        binds at 0.05 m; at 200 Hz the λ/5 cap (1.5 m) is the tighter one
        and nothing changes."""
        _, elastic = _fluid_and_elastic_envs()
        m = _stub_grid(monkeypatch, RAM(backend='rams',
                                        rams_dr_factor=1.0,
                           verbose=False), dr=10.0, dz=0.01)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dr_hi, _ = ram_grid.compute_grid_lytaev(elastic, 5000.0,
                                              max_range=5000.0, kind='rams',
                                              knobs=m._knob_record(),
                                              log=m._log,
                                              speed_bounds=m._speed_bounds)
            m = _stub_grid(monkeypatch, RAM(backend='rams',
                                            rams_dr_factor=1.0,
                               verbose=False), dr=10.0, dz=1.0)
            dr_lo, _ = ram_grid.compute_grid_lytaev(elastic, 200.0,
                                              max_range=5000.0, kind='rams',
                                              knobs=m._knob_record(),
                                              log=m._log,
                                              speed_bounds=m._speed_bounds)
        assert 0.04 < dr_hi < 1500.0 / (5.0 * 5000.0)
        assert dr_lo == pytest.approx(1500.0 / (5.0 * 200.0))

    def test_a_rotation_limited_case_takes_the_widest_stable_angle_or_refuses_a_pinned_one(
            self, monkeypatch):
        """In 4000 m of water at 1 kHz the rotated square root's own growth
        (3.7e-4 Np/m at 45°) exceeds what the seabed leaks, so no step
        helps. Left to itself the wrapper takes the widest stable angle
        (40°), and the resolution of the run's angles says so once — the
        grid chooser, which reads the same angle, stays silent about it; a
        pinned 45° is refused naming that angle."""
        deep = Environment(bathymetry=4000.0, ssp=1500.0,
                           bottom=_elastic_halfspace())
        m = _stub_grid(monkeypatch, RAM(backend='rams', verbose=False),
                       dr=10.0, dz=1.0)
        with recorded_warnings() as rec:
            dr, _ = ram_grid.compute_grid_lytaev(deep, 1000.0,
                                                 max_range=5000.0,
                                           kind='rams', knobs=m._knob_record(),
                                           log=m._log,
                                           speed_bounds=m._speed_bounds)
            assert ram_stability.resolve_rams_rotation_angle(
                deep, 1000.0, knobs=m._knob_record(),
                speed_bounds=m._speed_bounds) == 40.0
        assert not [w for w in rec if 'using rams_rotation_angle' in str(w.message)]
        with recorded_warnings() as rec:
            assert ram_stability.resolve_collins_thetas(
                deep, 'rams', [1000.0], knobs=m._knob_record(),
                speed_bounds=m._speed_bounds) == [40.0]
        lowered = [w for w in rec if 'using rams_rotation_angle=40' in str(w.message)]
        assert len(lowered) == 1
        assert 0.0 < dr < 0.3
        m = _stub_grid(monkeypatch, RAM(backend='rams', rams_rotation_angle=45.0,
                                        verbose=False),
                       dr=10.0, dz=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ConfigurationError, match='rams_rotation_angle=40'):
                ram_grid.compute_grid_lytaev(deep, 1000.0, max_range=5000.0,
                                       kind='rams', knobs=m._knob_record(),
                                       log=m._log,
                                       speed_bounds=m._speed_bounds)
        assert ram_stability.theta_for_freq(
            1000.0, knobs=RAM(backend='rams',
                              verbose=False)._knob_record()) == 45.0

    def test_rams_dz_is_capped_at_the_shear_wavelength(self, monkeypatch):
        from uacpy.models.pe_grid import rams_dz_shear_cap
        _, elastic = _fluid_and_elastic_envs()
        m = _stub_grid(monkeypatch, RAM(backend='rams', verbose=False),
                       dr=10.0, dz=1.0)
        _, dz = ram_grid.compute_grid_lytaev(elastic, 100.0, max_range=5000.0,
                                       kind='rams', knobs=m._knob_record(),
                                       log=m._log,
                                       speed_bounds=m._speed_bounds)
        assert dz == pytest.approx(rams_dz_shear_cap(400.0, 100.0))

    def test_depth_grid_is_capped_at_100000_points(self, monkeypatch):
        # dz = 0.5 mm over 100 m asks for 200000 depth points; the runtime
        # cap coarsens it to ~1 mm with a warning. f = 100 kHz keeps the
        # λ_p/16 stability floor (≈ 0.94 mm) below the capped value.
        fluid, _ = _fluid_and_elastic_envs()
        m = _stub_grid(monkeypatch, RAM(backend='ramgeo', verbose=False),
                       dr=10.0, dz=0.0005)
        from uacpy.models.ram._domain import MAX_DEPTH_POINTS
        from uacpy.models.ram.grid import SEAFLOOR_CELL_OFFSET
        assert MAX_DEPTH_POINTS == 100000
        with pytest.warns(UserWarning, match='100000') as record:
            _, dz = ram_grid.compute_grid_lytaev(fluid, 100000.0,
                                                 max_range=2000.0,
                                           kind='ramgeo',
                                           knobs=m._knob_record(), log=m._log,
                                           speed_bounds=m._speed_bounds)
        assert dz == pytest.approx(
            100.0 / (MAX_DEPTH_POINTS + SEAFLOOR_CELL_OFFSET), rel=1e-6)
        # The only way past the cap is a pinned grid: angle_max cannot lift
        # a floored dz, so the message does not offer it.
        assert not [w for w in record if 'angle_max' in str(w.message)]

    def test_the_point_cap_is_silent_where_the_floor_alone_binds(self,
                                                                 monkeypatch):
        """The cap is measured against the coarser of the optimiser's dz
        and the λ_p/16 floor. Here the same 5 mm request at 1 kHz is lifted
        to the 0.094 m floor and refined from it for the band's steepest
        component — about 0.05 m, far under the 100 000-point cap over
        100 m — so the cap never binds and a warning naming a 5 mm grid that
        never runs would be noise."""
        from uacpy.models.ram._domain import MAX_DEPTH_POINTS
        fluid, _ = _fluid_and_elastic_envs()
        m = _stub_grid(monkeypatch, RAM(backend='ramgeo', verbose=False),
                       dr=10.0, dz=0.005)
        with recorded_warnings() as caught:
            _, dz = ram_grid.compute_grid_lytaev(fluid, 1000.0,
                                                 max_range=2000.0,
                                           kind='ramgeo',
                                           knobs=m._knob_record(), log=m._log,
                                           speed_bounds=m._speed_bounds)
        assert not [w for w in caught if '100000' in str(w.message)]
        assert dz > 0.005 and 100.0 / dz < MAX_DEPTH_POINTS

    def test_the_rams_grid_log_does_not_call_the_pade_score_its_error(
            self, monkeypatch):
        """With ``rams_rotation=True`` rams0.5 marches a Crank-Nicolson step of the
        rotated square root (``rpade``), not the split-step Padé exponential
        the optimiser scores, so the grid line reports the score as what it
        is; with ``rams_rotation=False`` the scored operator is the marched one."""
        _, elastic = _fluid_and_elastic_envs()
        lines = {}
        for rotation in (True, False):
            m = _stub_grid(monkeypatch, RAM(backend='rams', rams_rotation=rotation,
                                            verbose=False),
                           dr=10.0, dz=1.0)
            logged = []
            m._log = lambda msg, level='info', _l=logged: _l.append(msg)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ram_grid.compute_grid_lytaev(elastic, 100.0, max_range=5000.0,
                                       kind='rams', knobs=m._knob_record(),
                                       log=m._log,
                                       speed_bounds=m._speed_bounds)
            lines[rotation] = [ln for ln in logged if 'Lytaev grid' in ln][0]
        assert 'predicted error' not in lines[True]
        assert 'split-step Padé score' in lines[True]
        assert 'predicted error' in lines[False]


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
            ram_collins.warn_if_ramsurf_crests(env)
        depths = dict(ram_grid.ramsurf_surface_nodes(env, 5000.0,
                                                     knobs=m._knob_record()))
        assert depths[500.0] == pytest.approx(0.0)     # crest clamped
        assert depths[1000.0] == pytest.approx(2.0)    # trough negated
        assert all(z >= 0.0 for z in depths.values())

    def test_all_depression_profile_is_silent(self):
        m = RAM(backend='ramsurf', verbose=False)
        env = self._env([(0.0, 0.0), (1000.0, -2.0), (5000.0, 0.0)])
        with recorded_warnings() as caught:
            ram_collins.warn_if_ramsurf_crests(env)
            surface = ram_grid.ramsurf_surface_nodes(env, 5000.0,
                                                     knobs=m._knob_record())
        assert not [w for w in caught if 'clamped' in str(w.message)]
        assert dict(surface)[1000.0] == pytest.approx(2.0)

    def test_missing_altimetry_is_a_configuration_error(self):
        m = RAM(backend='ramsurf', verbose=False)
        env = Environment(name='flat', bathymetry=100.0, ssp=1500.0)
        with pytest.raises(ConfigurationError, match='altimetry'):
            ram_grid.ramsurf_surface_nodes(env, 5000.0, knobs=m._knob_record())


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
            [near, far], ranges=np.array([0.0, 2000.0]))).dB)
        uniform = np.asarray(self._run(Bottom.from_columns(
            [near, near], ranges=np.array([0.0, 2000.0]))).dB)
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
    from uacpy.models.ram.mpirams import MPIRAMS_RANGE_TOL_M

    ranges = [500.0, 1500.0, 3000.0]
    receiver = Receiver(depths=[10.0, 20.0], ranges=ranges)
    stored = np.asarray(receiver.ranges, dtype=float)

    # The carrier forbids a range axis that is not strictly increasing, so
    # ranges[0] is the minimum.
    assert np.all(np.diff(stored) > 0)
    assert float(stored[0]) == float(stored.min())
    with pytest.raises(uacpy.ConfigurationError,
                       match='Receiver.ranges must be strictly increasing'):
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
        from uacpy.models.base import StageInputs
        model = RAM(backend='mpirams', verbose=False, dr=20.0, dz=2.0)
        env = TestMpiramsOutputRangeSpacing.ENV
        source = TestMpiramsOutputRangeSpacing.SOURCE
        # The settings of a receiver the checks accept, with the same
        # farthest range; the deck is then written for the grid under test,
        # which the refusal would stop before any deck exists.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            settings = model.run_settings(
                env, source, Receiver(depths=[20.0], ranges=[max(ranges)]))
        receiver = Receiver(depths=[20.0], ranges=ranges)
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            model._write_input(StageInputs(
                work_dir=work_dir, env=model._project_environment(env),
                source=source, receiver=receiver, settings=settings))
            model._run_binary(work_dir)
            return np.asarray(read_psif(work_dir).ranges, dtype=float)

    @pytest.mark.parametrize('run_mode', [RunMode.COHERENT_TL,
                                          RunMode.BROADBAND])
    def test_a_pair_below_the_exit_tolerance_is_refused(self, run_mode):
        """The message carries the two ranges, their separation and the limit
        — the three numbers needed to respace the grid."""
        receiver = self._receiver([500.0, 1000.0, 1000.05, 1500.0])
        model = RAM(backend='mpirams', verbose=False)
        with pytest.raises(ConfigurationError,
                           match='the mpiramS march resolves') as exc:
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
        RAM(backend='mpirams', verbose=False).validate_inputs(
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
        model = RAM(backend='mpirams', verbose=False)
        if collapses:
            with pytest.raises(ConfigurationError,
                               match='the mpiramS march resolves'):
                model.validate_inputs(self.ENV, self.SOURCE,
                                      self._receiver(ranges),
                                      run_mode=RunMode.COHERENT_TL)
        else:
            model.validate_inputs(self.ENV, self.SOURCE,
                                  self._receiver(ranges),
                                  run_mode=RunMode.COHERENT_TL)

    @pytest.mark.parametrize('run_mode', [None, 'coherent_tl',
                                          RunMode.COHERENT_TL])
    def test_validate_inputs_resolves_the_run_mode_as_run_does(self,
                                                               run_mode):
        """``None`` (the default) and the value string check as
        ``COHERENT_TL``, the mode ``run()`` resolves them to, so a
        two-frequency Source refuses on all three spellings; one frequency
        passes on all three."""
        model = RAM(backend='mpirams', verbose=False)
        receiver = self._receiver([500.0, 1000.0])
        two = Source(depths=50.0, frequencies=[100.0, 200.0])
        with pytest.raises(ConfigurationError, match='single source frequency'):
            model.validate_inputs(self.ENV, two, receiver, run_mode=run_mode)
        model.validate_inputs(self.ENV, self.SOURCE, receiver,
                              run_mode=run_mode)

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

    The bathymetry is deliberately non-monotonic so
    ``ram.collins.bathy_anchor_ranges`` emits many sections, and the layer
    thickness varies with range so the sediment base — where the absorbing ramp
    starts — moves between them.
    """
    ssp = SoundSpeedProfile(
        depths=np.array([0.0, 50.0, 200.0]),
        sound_speed=np.array([[1500.0, 1510.0],
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
        water_density=1.0,   # the recorded decks below took rho_w = 1
        name='rd-elastic',
        bathymetry=[(0.0, 120.0), (2000.0, 180.0), (4000.0, 90.0),
                    (6000.0, 200.0)],
        ssp=ssp, bottom=bottom,
    )


def _env_flat_fluid():
    return Environment(name='flat-fluid', bathymetry=100.0, ssp=1500.0,
                       water_density=1.0, bottom=_fluid_column(15.0, 1650.0))


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
    """``ram.collins.collins_deck_base`` +
    ``ram.collins.ramp_range_segments`` must reproduce
    ``ram.collins.collins_range_segments`` exactly at every frequency,
    because the broadband loop cuts the deck once for the whole band and
    re-ramps it."""

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
        base = ram_collins.collins_deck_base(env, kind, zmax,
                                             knobs=model._knob_record())
        for freq in sorted(_DECK_FREQS, reverse=True) + list(_DECK_FREQS):
            reramped = ram_collins.ramp_range_segments(
                env, base, freq, kind=kind, zmax=zmax,
                knobs=model._knob_record(), speed_bounds=model._speed_bounds)
            rebuilt = ram_collins.collins_range_segments(
                env, kind, zmax, freq, knobs=model._knob_record(),
                speed_bounds=model._speed_bounds)
            assert _exact(reramped) == _exact(rebuilt), f"freq={freq}"
            for segment in reramped:
                for value in segment.values():
                    if isinstance(value, list):
                        value.clear()

    def test_the_deck_matches_the_values_recorded_before_the_hoist(
            self, monkeypatch):
        """The independent pin: decks recorded from the single-stage builder
        the split replaced. Every other check here compares the two stages
        against a rebuild that now goes *through* them, so this is what
        stands between the deck and a shared drift.

        The attenuation ramp starts ``absorber_width_wavelengths`` wavelengths of
        the PE reference speed above ``zmax``, so that one value moves with
        the ``c0`` rule (1665 m/s here, ``reference_speed``); every other
        number is independent of it.

        The decks were recorded with the declared SSP profiles only; the
        intermediate profiles RAM now writes along a range-dependent SSP
        (``ram._domain.ssp_range_axis``) are pinned by
        ``TestRamFollowsTheLinearSspBetweenDeclaredProfiles``, so here the
        step allowance is lifted to keep the recorded sections."""
        monkeypatch.setattr(ram_domain, '_SSP_STEP_MAX_MPS', float('inf'))
        model = RAM(verbose=False, earth_curvature=False)

        flat = ram_collins.collins_range_segments(_env_flat_fluid(), 'rams',
                                             400.0, 800.0,
                                             knobs=model._knob_record(),
                                             speed_bounds=model._speed_bounds)
        assert _exact(flat) == [{
            'range': '0.0',
            'water_ssp': [('0.0', '1500.0'), ('100.0', '1500.0')],
            'bottom_c': [('100.0', '1650.0'), ('115.0', '1650.0'),
                         ('115.0', '1900.0'), ('400.0', '1900.0')],
            'bottom_rho': [('100.0', '1.6'), ('115.0', '1.6'),
                           ('115.0', '1.9'), ('400.0', '1.9')],
            'bottom_attn': [('100.0', '0.4'), ('115.0', '0.4'),
                            ('115.0', '0.2'),
                            ('358.37530555411473', '0.2'), ('400.0', '10.0')],
            'bottom_cs': [('100.0', '0.0'), ('115.0', '0.0'),
                          ('115.0', '0.0'), ('400.0', '0.0')],
            'bottom_attns': [('100.0', '0.0'), ('115.0', '0.0'),
                             ('115.0', '0.0'), ('400.0', '0.0')],
        }]

        sloped = ram_collins.collins_range_segments(
            _env_range_dependent_elastic(), 'ramgeo', 500.0, 50.0,
            knobs=model._knob_record(), speed_bounds=model._speed_bounds)
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

        # The 64-section elastic deck, spot-checked at both ends: full text
        # here would be unreadable, and the ends are where the seafloor is
        # shallowest and deepest. 62 sections re-anchor the layers to the
        # sloping seafloor; the other two open at the bottom's own column
        # switches, 1500 m and 4500 m, which fall inside anchor cells.
        wide = _exact(ram_collins.collins_range_segments(
            _env_range_dependent_elastic(), 'rams', 500.0, 200.0,
            knobs=model._knob_record(), speed_bounds=model._speed_bounds))
        assert len(wide) == 64
        switches = {seg['range']: seg['bottom_c'][0][1] for seg in wide
                    if seg['range'] in ('1500.0', '4500.0')}
        assert switches == {'1500.0': '1750.0', '4500.0': '1650.0'}
        assert wide[0]['range'] == '0.0'
        assert wide[-1]['range'] == '5973.6070381231675'
        assert wide[0]['bottom_attn'] == [
            ('120.0', '0.5'), ('140.0', '0.5'), ('140.0', '0.1'),
            ('334.87936798919645', '0.1'), ('500.0', '10.0')]
        assert wide[-1]['bottom_attn'] == [
            ('200.0', '0.5'), ('230.0', '0.5'), ('230.0', '0.1'),
            ('334.87936798919645', '0.1'), ('500.0', '10.0')]

    def test_the_attenuation_block_changes_length_with_frequency(self):
        """The block is not a fixed-shape array that could be scaled in
        place: the ramp keeps every control point at or above its start
        depth, and that count moves with ``absorbing_width ∝ 1/f``."""
        model = RAM(verbose=False)
        env = _env_range_dependent_elastic()
        lengths = {
            freq: {len(seg['bottom_attn'])
                   for seg in ram_collins.collins_range_segments(
                       env, 'rams', 500.0, freq, knobs=model._knob_record(),
                       speed_bounds=model._speed_bounds)}
            for freq in (50.0, 800.0)
        }
        assert lengths[50.0] == {4}
        assert lengths[800.0] == {5}

    def test_the_attenuation_block_changes_length_within_one_deck(self):
        """Worse than a per-frequency length: at one frequency the sections
        of a single deck disagree, because the seafloor — and with it the
        sediment base the ramp is clamped to — moves with range."""
        model = RAM(verbose=False)
        segments = ram_collins.collins_range_segments(
            _env_range_dependent_elastic(), 'rams', 500.0, 105.0,
            knobs=model._knob_record(), speed_bounds=model._speed_bounds)
        lengths = [len(seg['bottom_attn']) for seg in segments]
        assert set(lengths) == {4, 5}
        assert min(lengths.count(4), lengths.count(5)) > 10

    @pytest.mark.parametrize('kind,zmax', [('ramgeo', 501.0), ('rams', 500.0)])
    def test_a_deck_cut_against_another_grid_is_refused(self, kind, zmax):
        """Condition the hoist rests on: ``zmax`` is fixed for the band.
        Nothing in the code forces that, so reuse is checked rather than
        assumed — a stale deck would march a different domain in silence."""
        model = RAM(verbose=False)
        base = ram_collins.collins_deck_base(
            _env_range_dependent_elastic(), 'ramsurf', 500.0,
            knobs=model._knob_record())
        with pytest.raises(ConfigurationError, match='zmax'):
            ram_collins.ramp_range_segments(_env_range_dependent_elastic(),
                                            base,
                                       100.0, kind=kind, zmax=zmax,
                                       knobs=model._knob_record(),
                                       speed_bounds=model._speed_bounds)

    def test_the_broadband_loop_owns_one_deck_and_keeps_it_off_the_model(
            self, monkeypatch):
        """The deck is a local of the sweep, not a cache on ``self``: an
        attribute would outlive this ``env`` and be handed to the next
        ``run()``. Every bin must also get the *same* payload, at the
        band's own ``zmax``."""
        model = RAM(verbose=False, backend='ramgeo', dr=50.0, dz=0.5,
                    zmax=400.0, q_factor=4.0, record_duration=0.2)
        env = _env_slope_range_dependent_bottom()
        source = Source(depths=[40.0], frequencies=[150.0])
        receiver = Receiver(depths=np.array([30.0, 90.0]),
                            ranges=np.array([500.0, 1500.0]))

        class _Stop(Exception):
            pass

        seen, builds = [], []
        before = set(vars(model))
        real_base = ram_collins.collins_deck_base

        def spy_base(*args, **kwargs):
            built = real_base(*args, **kwargs)
            builds.append(built)
            return built

        def stub_launch(inputs):
            seen.append(inputs.prepared)
            if len(seen) == 3:
                raise _Stop

        monkeypatch.setattr(
            'uacpy.models.ram._model.collins_deck_base', spy_base)
        monkeypatch.setattr(
            'uacpy.models.ram.collins.collins_deck_base', spy_base)
        model._run_collins_binary = stub_launch
        monkeypatch.setattr(
            'uacpy.models.ram._model.read_collins_output',
            lambda inputs, **kwargs: {})
        with pytest.raises(_Stop):
            model.run(env, source, receiver, run_mode='broadband')

        # The decks are cut once for the band, with its water attenuation,
        # and every launch gets that one payload.
        decks = [b for b in builds if 'water_alpha' in b]
        assert len(decks) == 1
        assert all(deck is decks[0] for deck in seen)
        assert decks[0]['zmax'] == 400.0
        assert set(vars(model)) - before <= {'_run_collins_binary'}
        assert not any(value is decks[0] for value in vars(model).values())


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

        got = ram_collins.bathy_anchor_ranges(env, bottom)
        assert [repr(v) for v in got] == [repr(v) for v in expected]

    def test_the_sloping_case_keeps_its_recorded_anchors(self):
        """A second, independent pin: values recorded from the scalar-probe
        implementation, so a change to *both* paths still fails."""
        env = _env_slope_range_independent_bottom()
        got = ram_collins.bathy_anchor_ranges(env, env.bottom)
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
        from uacpy.models.pe_grid import rams_dz_shear_cap
        env = self._elastic_env()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = RAM(backend='rams', verbose=False)
            _, dz = ram_grid.compute_grid_lytaev(
                env, freq, max_range=5000.0, kind='rams',
                knobs=model._knob_record(), log=model._log,
                speed_bounds=model._speed_bounds)
        layers = 100.0 / dz
        assert layers == pytest.approx(round(layers), abs=1e-6), (
            f"h/dz = {layers} is not an integer, so the seafloor falls "
            f"between depth nodes")
        assert dz <= rams_dz_shear_cap(300.0, freq) * (1 + 1e-12), (
            "the aligned dz must still resolve the shear wavelength")

    def test_alignment_helper_respects_the_bound_direction(self):
        model = RAM(backend='rams', verbose=False)
        env = self._elastic_env()
        tighter = ram_grid.align_dz_with_seafloor(env, 0.428571, kind='rams',
                                                  knobs=model._knob_record())
        coarser = ram_grid.align_dz_with_seafloor(env, 0.428571, kind='rams',
                                                coarsen=True,
                                                knobs=model._knob_record())
        assert tighter <= 0.428571 < coarser
        for dz in (tighter, coarser):
            assert 100.0 / dz == pytest.approx(round(100.0 / dz), abs=1e-6)

    def test_an_already_aligned_dz_is_left_where_it_is(self):
        model = RAM(backend='rams', verbose=False)
        env = self._elastic_env()
        aligned = ram_grid.snap_dz_to_seafloor(100.0, 250)
        assert ram_grid.align_dz_with_seafloor(
            env, aligned, kind='rams',
            knobs=model._knob_record()) == pytest.approx(
            aligned, rel=1e-9)


class TestRamBackendBinaryResolution:
    """``RAM(backend='rams')`` never executes ``s_mpiram``, so resolving it at
    construction refuses a configuration that runs; a pinned
    ``executable`` is the binary of the named backend (mpiramS when
    ``backend=None``) and a dispatch to any other backend refuses."""

    @pytest.mark.parametrize('backend,expected', [
        ('rams', 'rams0.5'), ('ramsurf', 'ramsurf1.5'), ('ramgeo', 'ramgeo'),
        ('mpirams', 's_mpiram'), (None, 's_mpiram'),
    ])
    def test_the_resolved_binary_follows_the_backend(self, backend, expected):
        assert RAM(backend=backend, verbose=False)._exe.name == expected

    def test_an_unknown_backend_is_refused_before_any_lookup(self):
        with pytest.raises(ConfigurationError, match='not a known backend'):
            RAM(backend='definitely-not-a-backend', verbose=False)

    def test_a_pinned_executable_reaches_its_own_backend_only(self):
        exe = RAM(backend='rams', verbose=False)._exe
        model = RAM(backend='rams', executable=exe, verbose=False)
        assert model._collins_binary('rams') == exe
        with pytest.raises(ConfigurationError, match="dispatches to the 'ramgeo'"):
            model._collins_binary('ramgeo')

    def test_a_pinned_mpirams_binary_refuses_a_collins_dispatch(self):
        exe = RAM(verbose=False)._exe
        model = RAM(executable=exe, verbose=False)
        assert model._exe == exe
        with pytest.raises(ConfigurationError,
                           match="pins the mpirams binary.*backend='rams'"):
            model._collins_binary('rams')

    def test_a_relative_pinned_executable_launches_by_absolute_path(
            self, monkeypatch):
        """Every launch runs with ``cwd=`` a scratch directory, so a
        relative pin must reach the launch as an absolute path; the
        verbatim argument is kept for ``copy()`` / ``repr``."""
        import os
        exe = RAM(verbose=False)._exe
        monkeypatch.chdir(exe.parent.parent)
        rel = Path(exe.parent.name) / exe.name
        model = RAM(executable=rel, verbose=False)
        assert model.executable == rel
        assert model._exe.is_absolute()
        assert model._exe == exe.resolve()
        assert os.path.samefile(model._exe, exe)

    def test_a_pinned_missing_executable_is_named(self):
        from uacpy.core.exceptions import ExecutableNotFoundError
        with pytest.raises(ExecutableNotFoundError,
                           match='executable not found: /no/such/binary'):
            RAM(backend='rams', executable='/no/such/binary', verbose=False)


def test_ram_source_row_guard_runs_before_the_deck_is_written(tmp_path):
    """``ram._domain.check_source_row_is_solved`` refuses a source shallower
    than one depth cell; on the mpiramS path it used to fire after ssp.dat,
    the bathymetry file, the sediment file and ranges.dat were already on
    disk."""
    env = Environment(
        name='shallow_source', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1600.0, density=1.6,
                                  attenuation=0.5))
    source = Source(depths=0.05, frequencies=50.0)
    receiver = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
    model = RAM(backend='mpirams', verbose=False, dr=10.0, dz=1.0,
                work_dir=str(tmp_path / 'wd'))
    with pytest.raises(ConfigurationError, match='shallower than one depth'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.run(env, source, receiver)
    assert list(tmp_path.iterdir()) == [], (
        "the refusal left a half-written deck behind")


class TestRamHalfSpaceSedimentLayerIsTheAbsorberSpan:
    """``sedlayer`` is where mpiramS's attenuation ramp starts below the
    seafloor (:func:`~uacpy.models.ram._domain.absorber_span`), floored at one
    depth cell so control point ``nzs-1`` sits below the seafloor point. A bare
    half-space adds no layer thickness of its own: the span alone sets it, on
    every grid.
    """

    @staticmethod
    def _halfspace_env(depth):
        return Environment(
            bathymetry=depth, ssp=1500.0,
            bottom=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0, density=1.8,
                attenuation=0.5, shear_speed=800.0, shear_attenuation=0.2))

    def _sedlayer(self, depth, span, zmax, dz, tmp_path):
        model = RAM(backend='rams', verbose=False)
        return ram_mpirams.prepare_bottom_properties(
            self._halfspace_env(depth), tmp_path, span, zmax, dz=dz,
            knobs=model._knob_record(), log=model._log)[0]

    @pytest.mark.parametrize('depth', [100.0, 30.0, 3000.0])
    def test_the_span_sets_the_thickness_at_every_depth(self, depth,
                                                        tmp_path):
        # No depth-fraction floor: 1 m of span stays 1 m in 3 km of water.
        assert self._sedlayer(depth, 1.0, depth * 2.0, 0.5, tmp_path) == \
            pytest.approx(1.0)

    def test_a_span_thinner_than_one_cell_is_floored_at_the_cell(self,
                                                                 tmp_path):
        assert self._sedlayer(100.0, 0.2, 200.0, 0.5, tmp_path) == \
            pytest.approx(0.5)

    def test_a_span_of_exactly_one_cell_is_the_cell(self, tmp_path):
        assert self._sedlayer(100.0, 0.5, 200.0, 0.5, tmp_path) == \
            pytest.approx(0.5)

    def test_a_deeper_absorber_span_sets_the_thickness(self, tmp_path):
        assert self._sedlayer(100.0, 250.0, 200.0, 0.5, tmp_path) == \
            pytest.approx(250.0)

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
            dr = ram_collins.constrain_dr_to_sections(
                250.0, self.SEGMENTS, pinned=False, log=RAM(
                    backend='ramgeo', verbose=False)._log)
        assert dr == pytest.approx(100.0 * (1 - 1e-4))

    def test_a_callers_own_dr_is_bounded_and_announced(self):
        with pytest.warns(UserWarning, match='profile-section spacing'):
            dr = ram_collins.constrain_dr_to_sections(
                250.0, self.SEGMENTS, pinned=True, log=RAM(backend='ramgeo',
                                                           verbose=False)._log)
        assert dr == pytest.approx(100.0 * (1 - 1e-4))   # bounded either way

    def test_a_dr_already_inside_the_spacing_is_left_alone(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            dr = ram_collins.constrain_dr_to_sections(
                50.0, self.SEGMENTS, pinned=True, log=RAM(backend='ramgeo',
                                                          verbose=False)._log)
        assert dr == pytest.approx(50.0)


class TestCollinsBroadbandSizesItsRangeStepAtTheBandTop:
    """A broadband march runs one grid for the whole band, so the step has to
    resolve the SMALLEST wavelength in it. Sizing ``dr`` at ``freq_min`` leaves it
    ``freq_max/freq_min`` times too coarse at the top: measured on a 100 m Pekeris
    guide (1700/1.7/0.5) against Scooter at 500 Hz, ramgeo with ``dr`` from
    ``freq_min = 100 Hz`` (54.05 m) is 8.85 dB rms / 16.86 dB max out, against
    1.64 / 5.24 with ``dr`` from ``freq_max`` (12.81 m).

    ``rams`` keeps ``freq_min`` deliberately, and that is not the same question:
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
        sized at freq_min the broadband bin was marching a step 4x too coarse for
        it, and the two disagreed by 7 dB rms.
        """
        import numpy as np
        from uacpy.core import Receiver, Source
        from uacpy.models import RAM
        from uacpy.core.run_settings import RunMode
        env = self._env()
        ranges = np.linspace(500.0, 5000.0, 19)
        rcv = Receiver(depths=[50.0], ranges=ranges)
        band = np.linspace(100.0, 500.0, 9)
        wide = RAM(backend='ramgeo', verbose=False).run(
            env, Source(depths=25.0, frequencies=band), rcv,
            run_mode=RunMode.BROADBAND)
        dB = np.asarray(wide.dB, dtype=float)
        top = dB[..., -1].ravel() if dB.ndim > 1 else dB.ravel()
        narrow = np.asarray(RAM(backend='ramgeo', verbose=False).run(
            env, Source(depths=25.0, frequencies=500.0), rcv).dB,
            dtype=float).ravel()
        rms = float(np.sqrt(np.nanmean((top - narrow) ** 2)))
        assert rms < 2.0, f"broadband top bin is {rms:.2f} dB rms from the " \
                          f"narrowband run at the same frequency"


class TestRamWarnsWhenReceiverRangesStartAtZero:
    """RAM writes no r=0 column on any backend: the self-starter sets ``r=dr``
    before its only ``outpt`` call (``ramgeo1.5.f:124,172``) and the march loop
    advances ``r=r+dr`` at ``:82`` BEFORE ``outpt`` at ``:83``. Every backend
    hands its assembled field to the shared ``_mask_source_axis``, which NaNs
    that column and warns once per run with the family's one wording — so
    ``np.linspace(0, R, N)`` reads the same on RAM as on Kraken or OASES.
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
        return [w for w in record if 'at r = 0, where' in str(w.message)]

    def test_a_zero_first_range_warns_and_the_column_is_nan(self):
        import numpy as np
        from uacpy.core import Receiver, Source
        from uacpy.models import RAM
        rcv = Receiver(depths=[50.0], ranges=np.array([0.0, 500.0, 1000.0]))
        with recorded_warnings() as rec:
            field = RAM(verbose=False).run(
                self._env(), Source(depths=25.0, frequencies=100.0), rcv)
        assert len(self._zero_range_warnings(rec)) == 1
        tl = np.asarray(field.to_dB().dB, dtype=float).ravel()
        assert np.isnan(tl[0]) and np.isfinite(tl[1:]).all()

    def test_a_positive_first_range_is_silent(self):
        import numpy as np
        from uacpy.core import Receiver, Source
        from uacpy.models import RAM
        rcv = Receiver(depths=[50.0], ranges=np.array([500.0, 1000.0]))
        with recorded_warnings() as rec:
            RAM(verbose=False).run(
                self._env(), Source(depths=25.0, frequencies=100.0), rcv)
        assert self._zero_range_warnings(rec) == []

    def test_a_broadband_sweep_warns_once_not_once_per_frequency(self):
        # The mask runs once on the assembled broadband field, not inside
        # the frequency loop.
        import numpy as np
        from uacpy.core import Receiver, Source
        from uacpy.models import RAM
        from uacpy.core.run_settings import RunMode
        rcv = Receiver(depths=[50.0], ranges=np.array([0.0, 500.0, 1000.0]))
        with recorded_warnings() as rec:
            RAM(verbose=False).run(
                self._env(),
                Source(depths=25.0, frequencies=[80.0, 100.0, 120.0]),
                rcv, run_mode=RunMode.BROADBAND)
        assert len(self._zero_range_warnings(rec)) == 1

    def test_the_collins_family_warns_on_the_same_geometry(self):
        # mpiramS and the Collins backends assemble their fields in different
        # methods; each hands its field to the same mask.
        import numpy as np
        from uacpy.core import Receiver, Source
        from uacpy.models import RAM
        rcv = Receiver(depths=[50.0], ranges=np.array([0.0, 500.0, 1000.0]))
        with recorded_warnings() as rec:
            RAM(backend='ramgeo', verbose=False).run(
                self._env(), Source(depths=25.0, frequencies=100.0), rcv)
        hits = self._zero_range_warnings(rec)
        assert len(hits) == 1
        assert str(hits[0].message).startswith(
            'RAM: 1 receiver range(s) at r = 0, where the point-source')


class TestRamLeavesRealSeabedBeforeTheAbsorber:
    """``ram._domain.absorber_span`` puts the artificial attenuation ramp over
    the deepest ``absorber_width_wavelengths`` wavelengths, so the
    NON-absorbing sub-bottom the automatic grid leaves is ``(zmax - depth) -
    absorbing_width``. With the old ``depth + dz + absorbing_width`` that was
    exactly ``dz`` — 1.0 m at 25 Hz, 0.31 m at 300 Hz — i.e. the ramp began one
    cell below the seafloor, which is the very thing
    ``ram._domain.absorber_span``'s own docstring warns against.

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
        c0 = ram_domain.resolve_c0(env, knobs=model._knob_record(),
                                   speed_bounds=model._speed_bounds)
        absorbing = model.absorber_width_wavelengths * c0 / freq
        return (ram_domain.adequate_zmax(
            env, freq, max_range=5000.0, knobs=model._knob_record(),
            speed_bounds=model._speed_bounds) - env.depth) - absorbing

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
        assert ram_domain.seabed_sound_speed(self._half_space(), 1500.0) == \
            pytest.approx(1800.0)

    def test_a_layered_bottom_pads_below_the_whole_stack(self):
        # The pad sits BELOW the modelled stack, never inside it. This used to
        # return one depth cell on a layered bottom so that the span mpiramS
        # spreads its sediment control points over would stay tight — which
        # left the stack itself outside the grid (measured 16.15 dB max on
        # ramgeo). The control-point interval is now held to one depth cell by
        # ``nzs`` instead, so the span is free to hold the seabed.
        from uacpy.models import RAM
        model = RAM(verbose=False)
        env = self._layered()
        stack = env.bottom.total_thickness_max()
        assert stack == pytest.approx(5.0)
        got = self._real_seabed(model, env, 100.0)
        pad = max(2.0 * 2800.0 / 100.0,
                  ram_domain.leaky_field_depth(env, 100.0, 5000.0,
                                               knobs=model._knob_record()))
        assert got == pytest.approx(stack + pad)


class TestRamGridHoldsTheWholeSedimentStack:
    """The automatic PE domain has to end BELOW the deepest modelled layer.

    ``ram._domain.adequate_zmax`` used to size the layered grid as ``depth + dz
    + absorbing_width``, with no term for the sediment stack, and nothing
    downstream clips to it: ``ram._seabed.piecewise_breakpoints`` takes
    ``zmax``
    only to give the half-space a non-zero depth extent and emits every layer
    step regardless, so the block runs past the grid floor, and ``zread``
    interpolates it onto the shorter grid, and on rams0.5 the fill loop
    replaces the layer with a linear gradient to the half-space value. The
    absorbing ramp disappears with it, because
    ``ram.collins.ramp_absorbing_attenuation`` returns the block unchanged once
    ``z_abs >= z_bottom``.

    Measured on 100 m of water over a 60 m layer (1600 m/s, 1.6, 0.05
    dB/lambda) on an 1800 / 2.0 / 0.1 half-space at 800 Hz, source 50 m, over
    200 m-2 km and 9 receiver depths: the automatic grid ended at 143.33 m
    against a layer base at 160 m and sat 2.75 dB rms / 15.77 dB max from a
    converged zmax=400 m on ramgeo, with no warning. An independent pass put
    the same defect at 3.37 / 16.15 on ramgeo over a wider sweep, 3.65 / 11.89
    on mpiramS and 3.65 / 9.13 on rams (200 Hz, 200 m layer) — silent on all
    three.
    """

    FREQ = 800.0

    @staticmethod
    def _stacked():
        return Environment(
            name='stack', bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=60.0, sound_speed=1600.0,
                                      density=1.6, attenuation=0.05)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1800.0,
                    density=2.0, attenuation=0.1)))

    def _absorbing_width(self, model, env):
        c0 = ram_domain.resolve_c0(env, knobs=model._knob_record(),
                                   speed_bounds=model._speed_bounds)
        return model.absorber_width_wavelengths * c0 / self.FREQ

    def test_the_automatic_grid_ends_below_the_deepest_layer_base(self):
        model = RAM(backend='ramgeo', verbose=False)
        env = self._stacked()
        base = env.depth + env.bottom.total_thickness_max()
        assert base == pytest.approx(160.0)
        assert ram_domain.compute_zmax(env, self.FREQ,
                                       max_range=5000.0, knobs=model._knob_record(),
                                       speed_bounds=model._speed_bounds) > base

    def test_the_absorbing_layer_starts_below_the_stack(self):
        # ram.pdf p.7 puts the ramp over "the lower few wavelengths of the
        # grid", not over the seabed: the whole stack has to sit above it.
        model = RAM(backend='ramgeo', verbose=False)
        env = self._stacked()
        zmax = ram_domain.compute_zmax(env, self.FREQ,
                                       max_range=5000.0, knobs=model._knob_record(),
                                       speed_bounds=model._speed_bounds)
        ramp_start = zmax - self._absorbing_width(model, env)
        assert ramp_start > env.depth + env.bottom.total_thickness_max()

    def test_a_range_dependent_stack_is_sized_by_its_deepest_column(self):
        # ``total_thickness_max`` reduces over columns, so a stack that
        # thickens with range still ends inside the grid at its deepest point.
        env = self._stacked()
        thick = SeabedColumn(
            layers=[SedimentLayer(thickness=140.0, sound_speed=1600.0,
                                  density=1.6, attenuation=0.05)],
            halfspace=env.bottom.columns[0].halfspace)
        rd = Environment(
            name='rd-stack', bathymetry=100.0, ssp=1500.0,
            bottom=Bottom(columns=[env.bottom.columns[0], thick],
                          ranges=[0.0, 5000.0]))
        model = RAM(backend='ramgeo', verbose=False)
        assert ram_domain.compute_zmax(
            rd, self.FREQ, max_range=5000.0, knobs=model._knob_record(),
            speed_bounds=model._speed_bounds) > 100.0 + 140.0

    def test_a_pinned_zmax_inside_the_stack_names_the_layer_base(self):
        model = RAM(backend='ramgeo', verbose=False, zmax=150.0)
        with pytest.warns(UserWarning, match='inside the sediment stack'):
            ram_domain.compute_zmax(self._stacked(), self.FREQ,
                                    max_range=5000.0, knobs=model._knob_record(),
                                    speed_bounds=model._speed_bounds)

    def test_a_pinned_zmax_below_the_stack_is_silent_about_it(self):
        model = RAM(backend='ramgeo', verbose=False, zmax=400.0)
        with recorded_warnings() as caught:
            ram_domain.compute_zmax(self._stacked(), self.FREQ,
                                    max_range=5000.0, knobs=model._knob_record(),
                                    speed_bounds=model._speed_bounds)
        assert not any('sediment stack' in str(w.message) for w in caught)

    def test_a_deep_half_space_pads_the_same_on_every_backend(
            self):
        """The pad below a bare half-space is the deeper of
        ``_SEABED_WAVELENGTHS_BEFORE_ABSORBER`` bottom wavelengths and the
        continuous spectrum's reach (``leaky_field_depth``), not a fraction of
        the water depth, and the mpiramS domain is the Collins domain snapped
        onto its ``dz``."""
        from uacpy.models.ram._domain import (
            _SEABED_WAVELENGTHS_BEFORE_ABSORBER,
        )
        model = RAM(backend='ramgeo', verbose=False)
        env = Environment(
            name='deep-halfspace', bathymetry=3000.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=2.0,
                                      attenuation=0.1))
        pad = max(_SEABED_WAVELENGTHS_BEFORE_ABSORBER * 1800.0 / self.FREQ,
                  ram_domain.leaky_field_depth(env, self.FREQ, 5000.0,
                                               knobs=model._knob_record()))
        collins = ram_domain.compute_zmax(env, self.FREQ,
                                          max_range=5000.0, knobs=model._knob_record(),
                                          speed_bounds=model._speed_bounds)
        assert collins - env.depth - self._absorbing_width(model, env) == \
            pytest.approx(pad)
        dz = 0.1
        mpirams = ram_mpirams.mpirams_zmax(env, self.FREQ, dz,
                                           max_range=5000.0, knobs=model._knob_record(),
                                           log=model._log,
                                           speed_bounds=model._speed_bounds)
        assert mpirams == pytest.approx(collins, abs=0.5 * dz + 1e-2)

def _mpirams_model():
    try:
        return uacpy.RAM(backend='mpirams', verbose=False)
    except ExecutableNotFoundError:
        pytest.skip("mpiramS binary not installed")


def _assemble(model, rout, ranges, tmp_path):
    """Drive ``ram.mpirams.assemble_tl_field`` over a synthesised mpiramS
    result whose ``rout`` is whatever the caller wants — including shorter
    than the ranges that were requested, which is what a march that stopped
    early would report."""
    from uacpy.io import PsifFile
    env = uacpy.Environment(bathymetry=200.0, ssp=1500.0)
    source = uacpy.Source(depths=50.0, frequencies=100.0)
    receiver = uacpy.Receiver(depths=np.array([25.0, 75.0]),
                              ranges=np.asarray(ranges, dtype=float))
    zg = np.linspace(0.0, 400.0, 81)
    rout = np.asarray(rout, dtype=float)
    # The header scalars play no part in the TL assembly.
    result = PsifFile(
        n_samples=0.0, c0=1500.0, water_min=1500.0, sample_rate=0.0,
        q_factor=0.0,
        frequencies=np.array([100.0]), depths=zg, ranges=rout,
        pe_field=np.full((zg.size, 1, rout.size), 1e-3,
                         dtype=np.complex128))
    return ram_mpirams.assemble_tl_field(
        result, env, source, receiver, Path(tmp_path), 100.0, 10.0,
        attach_output_paths=model._attach_output_paths,
        knobs=model._knob_record(), mask_source_axis=model._mask_source_axis,
        result_kwargs=model._result_kwargs)


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
    """``stability_range_m`` names an absolute range; on mpiramS it acts as one
    only when the receiver carries a single range.

    ``mpiramS/src/ram.f90:68`` sets ``rsc = |rg(nr)| - rs`` once, and
    ``:251`` tests ``abs(rend - rnow) < rsc`` where ``:166`` has reassigned
    ``rend`` to the *current* output range. The Collins codes test the
    absolute range instead (``ramgeo/ramgeo1.5.f:368``: ``if(r.ge.rs)``).
    """

    RANGES = list(np.linspace(500.0, 20000.0, 40))

    @staticmethod
    def _env():
        return make_pekeris(density=1.7)

    @staticmethod
    def _run(ranges, rs):
        kw = dict(backend='mpirams', verbose=False)
        if rs is not None:
            kw['stability_range_m'] = rs
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = RAM(**kw).run(
                TestRsStabilityIsInertOnAMultiRangeOutputGrid._env(),
                Source(depths=25.0, frequencies=100.0),
                Receiver(depths=[20.0, 50.0, 80.0], ranges=ranges))
        return np.asarray(field.data)

    def test_a_pinned_stability_range_warns_on_a_multi_range_grid(self):
        with pytest.warns(UserWarning, match='stability_range_m'):
            RAM(backend='mpirams', stability_range_m=10000.0, verbose=False).run(
                self._env(), Source(depths=25.0, frequencies=100.0),
                Receiver(depths=[50.0], ranges=[500.0, 20000.0]))

    def test_two_output_ranges_are_already_enough_to_warn(self):
        """The low side of the grid-size threshold."""
        model = RAM(backend='mpirams', stability_range_m=10000.0, verbose=False)
        with pytest.warns(UserWarning, match='stability_range_m'):
            ram_stability.warn_stability_range_inert_on_a_multi_range_grid(
                Receiver(depths=[50.0], ranges=[500.0, 20000.0]),
                knobs=model._knob_record())

    def test_a_single_output_range_does_not_warn(self):
        """The other side: with one output range ``rend`` never moves and the
        test reduces to the absolute one the parameter name promises."""
        model = RAM(backend='mpirams', stability_range_m=10000.0, verbose=False)
        with recorded_warnings() as caught:
            ram_stability.warn_stability_range_inert_on_a_multi_range_grid(
                Receiver(depths=[50.0], ranges=[20000.0]),
                knobs=model._knob_record())
        assert not [w for w in caught if 'stability_range_m' in str(w.message)]

    def test_an_unpinned_stability_range_does_not_warn(self):
        model = RAM(backend='mpirams', verbose=False)
        with recorded_warnings() as caught:
            ram_stability.warn_stability_range_inert_on_a_multi_range_grid(
                Receiver(depths=[50.0], ranges=self.RANGES),
                knobs=model._knob_record())
        assert not [w for w in caught if 'stability_range_m' in str(w.message)]

    @pytest.mark.slow
    def test_the_measurement_the_warning_reports(self):
        """Two ``stability_range_m`` values 9 km apart give bit-identical output on
        a 40-range grid and different output on a single-range one — the
        second half is what shows the observable is live rather than the
        fixture being blind to ``stability_range_m`` altogether."""
        multi = [self._run(self.RANGES, rs) for rs in (10000.0, 19000.0)]
        assert np.array_equal(multi[0], multi[1])
        single = [self._run([20000.0], rs) for rs in (10000.0, 19000.0)]
        assert not np.array_equal(single[0], single[1])


class TestRamOutputDomainStopsAtTheSeafloor:
    """RAM's output domain is the water column, and one number expresses it.

    ``core.bathymetry.mask_below_seafloor`` returns NaN below the seafloor although
    the PE marches through the sediment and computes a field there — measured
    on this fixture (mask neutralised at runtime): |p| at 120 m is 5.4e-4 on
    mpiramS and 6.8e-4 on ramgeo, against Kraken 6.3e-4 and Scooter 6.2e-4. The
    same depth gates source placement, so a source in the sediment is refused
    on RAM alone. These pin the three halves of that: the depth, the
    explanation the refusal carries, and the warning that must keep
    accompanying the receiver NaNs.
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
        return make_pekeris(density=1.7)

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
        with pytest.raises(InvalidDepthError,
                           match='exceeds resolvable depth') as exc:
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
        with pytest.raises(InvalidDepthError,
                           match='exceeds resolvable depth') as exc:
            RAM(verbose=False).validate_inputs(
                self._flat_env(),
                Source(depths=120.0, frequencies=100.0),
                Receiver(depths=[50.0], ranges=[2000.0]))
        assert 'output convention' not in str(exc.value)

    def test_a_below_seafloor_receiver_warns_and_comes_back_nan(self):
        """The NaN below the seafloor stays a *warned* NaN. Widening the cut
        to the media column would accept the buried source and silence this
        warning while ``core.bathymetry.mask_below_seafloor`` kept returning NaN
        — trading a warned NaN for a silent one."""
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


def _anchor_ranges_via_copies(env, bottom):
    """``ram.collins.bathy_anchor_ranges`` as it read the bottom before the
    fix.

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
                             [wide_range_dependent_env,
                              range_independent_layered_env],
                             ids=['range-dependent', 'range-independent'])
    def test_it_produces_the_doubles_the_copying_read_produced(self,
                                                               env_factory):
        env = env_factory()
        got = ram_collins.bathy_anchor_ranges(env, env.bottom)
        assert got == _anchor_ranges_via_copies(env, env.bottom)

    @pytest.mark.parametrize('env_factory',
                             [wide_range_dependent_env,
                              range_independent_layered_env],
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
        assert ram_collins.bathy_anchor_ranges(env, env.bottom) == \
            pytest.approx(_ANCHORS_WIDE if env.name == 'wide-rd'
                          else _ANCHORS_RI)

    def test_a_bottom_with_no_sediment_yields_no_anchors(self):
        env = Environment(
            name='halfspace', bathymetry=[(0.0, 60.0), (5000.0, 400.0)],
            ssp=1500.0,
            bottom=SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0, density=1.9,
                attenuation=0.2)))
        assert ram_collins.bathy_anchor_ranges(env, env.bottom) == []


# The anchor ranges the two fixtures produce, captured once through the
# copying reference implementation so the monkeypatched test above has
# something to compare against that ``Bottom.at`` did not compute.
_ANCHORS_WIDE = _anchor_ranges_via_copies(wide_range_dependent_env(),
                                          wide_range_dependent_env().bottom)


_ANCHORS_RI = _anchor_ranges_via_copies(range_independent_layered_env(),
                                        range_independent_layered_env().bottom)


def test_the_captured_anchor_lists_are_not_empty():
    # An anti-vacuity floor on the comparison above: two empty lists compare
    # equal, and the monkeypatched test would then assert nothing.
    assert len(_ANCHORS_WIDE) > 10
    assert len(_ANCHORS_RI) > 10


# ─── The water-SSP block ends at the grid floor ───────────────────────────


def _collins_launch_grid(model, env, src, rcv, run_mode=RunMode.COHERENT_TL):
    """The one Collins launch of ``model.run(env, src, rcv, run_mode)``,
    driven through the stage hooks on the settings ``run_settings``
    resolves, and its output on the binary's own grid
    (``ram.collins.read_collins_grid``)."""
    from uacpy.models.base import StageInputs
    settings = model.run_settings(env, src, rcv, run_mode)
    projected = model._project_environment(env)
    fm = model._setup_file_manager()
    try:
        inputs = StageInputs(
            work_dir=fm.work_dir, env=projected, source=src, receiver=rcv,
            settings=settings)
        deck = model._write_input(inputs)
        model._launch(inputs, deck)
        return ram_collins.read_collins_grid(inputs,
                                             knobs=model._knob_record(),
                                             speed_bounds=model._speed_bounds)
    finally:
        fm.finish()


def _halfspace_env(depth, ssp):
    return Environment(
        bathymetry=depth, ssp=ssp,
        bottom=Bottom.from_halfspace(BoundaryProperties(
            sound_speed=1700.0, density=1.8, attenuation=0.5)))


class TestWaterSspBlockEndsAtTheGridFloor:
    """``zread`` writes each water-SSP sample at node ``1.5 + z/dz`` with no
    bound (``ramgeo1.5.f:209-240``): a sample past ``mz`` overruns the array
    and one past ``nz+2`` becomes the value the fill loop ramps the whole
    column towards. The deck therefore ends the block at ``zmax`` with the
    profile's interpolated value there, exactly as the bottom blocks are
    cut."""

    def _deck(self, ssp, zmax, depth=50.0):
        model = RAM(verbose=False, earth_curvature=False)
        base = ram_collins.collins_deck_base(_halfspace_env(depth, ssp),
                                             'ramgeo',
                                        zmax, knobs=model._knob_record())
        return base['segments'][0]['water_ssp']

    def test_a_sample_below_zmax_is_replaced_by_the_value_at_zmax(self):
        rows = self._deck([(0.0, 1500.0), (2000.0, 1600.0)], zmax=100.0)
        assert rows == [(0.0, 1500.0), (100.0, 1505.0)]
        assert rows[-1][0] <= 100.0

    def test_a_profile_inside_the_grid_is_written_unchanged(self):
        rows = self._deck([(0.0, 1500.0), (30.0, 1490.0), (50.0, 1495.0)],
                          zmax=100.0)
        assert rows == [(0.0, 1500.0), (30.0, 1490.0), (50.0, 1495.0)]

    def test_a_sample_exactly_at_zmax_is_kept_and_nothing_is_appended(self):
        rows = self._deck([(0.0, 1500.0), (100.0, 1510.0), (200.0, 1520.0)],
                          zmax=100.0)
        assert rows == [(0.0, 1500.0), (100.0, 1510.0)]

    def test_a_two_point_profile_marches_like_its_hand_clipped_twin(self):
        """Same physics, two spellings: a profile tabulated to 2000 m over a
        50 m column, and the same profile cut by hand at the grid floor. The
        binary must see one deck for both."""
        src = Source(depths=20.0, frequencies=100.0)
        rcv = Receiver(depths=[20.0, 40.0], ranges=[500.0, 1000.0])

        def tl(ssp):
            model = RAM(backend='ramgeo', verbose=False, dz=0.5, dr=10.0,
                        zmax=100.0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return np.asarray(model.run(
                    _halfspace_env(50.0, ssp), src, rcv,
                    run_mode=RunMode.COHERENT_TL).dB, dtype=float)

        deep = tl([(0.0, 1500.0), (2000.0, 1600.0)])
        clipped = tl([(0.0, 1500.0), (100.0, 1505.0)])
        assert np.all(np.isfinite(deep))
        assert float(np.max(np.abs(deep - clipped))) < 0.05


# ─── The output stride resolves the modal beat ────────────────────────────


class TestOutputStrideResolvesTheModalBeat:
    """The modulus of the field beats at up to ``Δk = 2πf(1/c_min −
    1/c_max)``; ``ram._interp.interp_envelope_to_receiver_grid`` reads it
    between written ranges, so the output spacing ``dr·ndr`` is held at or
    below the beat period over ``COLLINS_SAMPLES_PER_BEAT`` — the one
    spacing rule the automatic ``dr`` obeys too. On the 1 kHz Pekeris
    reference the stride moved point TL by 0.7 dB between ``ndr`` 1 and 2 on
    one and the same march."""

    DR, RMAX, RANGES = 8.543, 20000.0, [5000.0, 10000.0, 15000.0, 20000.0]

    def test_the_pekeris_beat_wavenumber(self):
        env = _halfspace_env(100.0, 1500.0)
        dk = ram_domain.modal_beat_wavenumber(
            env, 1000.0, speed_bounds=RAM(verbose=False)._speed_bounds)
        assert dk == pytest.approx(
            2 * np.pi * 1000.0 * (1 / 1500.0 - 1 / 1700.0), rel=1e-12)
        assert dk == pytest.approx(0.493, abs=0.001)

    def test_a_beat_shorter_than_the_stride_lowers_ndr(self):
        ndr_count, _ = ram_collins.collins_output_stride(self.DR, self.RMAX,
                                                  self.RANGES)
        ndr_beat, _ = ram_collins.collins_output_stride(self.DR, self.RMAX,
                                                 self.RANGES,
                                                 beat_wavenumber=0.493)
        assert ndr_count == 2
        assert ndr_beat == 1

    def test_the_cap_sits_exactly_at_the_sampling_spacing(self):
        # 2π/(Δk·COLLINS_SAMPLES_PER_BEAT) = 2·dr: two steps per record still
        # sample the beat that often; a hair shorter does not.
        from uacpy.models.ram._domain import COLLINS_SAMPLES_PER_BEAT
        at = 2 * np.pi / (COLLINS_SAMPLES_PER_BEAT * 2 * self.DR)
        assert ram_collins.collins_output_stride(
            self.DR, self.RMAX, self.RANGES, beat_wavenumber=at)[0] == 2
        assert ram_collins.collins_output_stride(
            self.DR, self.RMAX, self.RANGES,
            beat_wavenumber=at * (1 + 1e-9))[0] == 1

    def test_the_record_count_ceiling_caps_ndr_above_the_beat_cap(self):
        ndr, _ = ram_collins.collins_output_stride(
            0.5, 200_000.0, [1.0, 200_000.0], beat_wavenumber=10.0)
        assert 200_000.0 / (0.5 * ndr) <= 20_000 + 1

    def test_an_isovelocity_environment_has_no_beat_cap(self):
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=Bottom.from_halfspace(BoundaryProperties(
                              sound_speed=1500.0, density=1.0,
                              attenuation=0.0)))
        assert ram_domain.modal_beat_wavenumber(
            env, 1000.0, speed_bounds=RAM(verbose=False)._speed_bounds) == 0.0
        assert ram_collins.collins_output_stride(
            self.DR, self.RMAX, self.RANGES, beat_wavenumber=0.0)[0] == 2


# ─── The automatic dz is capped at the source depth ───────────────────────


class TestAutoDzIsCappedAtTheSourceDepth:
    """Every binary plants the source at row ``1 + zs/dz`` and never solves
    row 1, so the depth cell can be no deeper than the source. The cost floor
    ``c_min/(16 f)`` is 10 m at 10 Hz; a 5 m source needs the automatic grid
    to come down to it, not a refusal."""

    ENV = _halfspace_env(100.0, 1500.0)

    @pytest.mark.parametrize('kind', ['ramgeo', 'mpirams'])
    def test_the_grid_comes_down_to_the_source(self, kind):
        model = RAM(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, dz_free = ram_grid.compute_grid_lytaev(
                self.ENV, 10.0, max_range=1000.0, kind=kind,
                knobs=model._knob_record(), log=model._log,
                speed_bounds=model._speed_bounds)
            _, dz_capped = ram_grid.compute_grid_lytaev(
                self.ENV, 10.0, max_range=1000.0, kind=kind, zs=5.0,
                knobs=model._knob_record(), log=model._log,
                speed_bounds=model._speed_bounds)
        assert dz_free > 5.0
        assert dz_capped <= 5.0
        from uacpy.models.ram.grid import SEAFLOOR_CELL_OFFSET
        layers = 100.0 / dz_capped - SEAFLOOR_CELL_OFFSET
        assert layers == pytest.approx(round(layers), abs=1e-9)
        ram_domain.check_source_row_is_solved(5.0, dz_capped)

    def test_a_source_below_one_cell_leaves_the_grid_alone(self):
        model = RAM(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, dz_free = ram_grid.compute_grid_lytaev(
                self.ENV, 10.0, max_range=1000.0, kind='ramgeo',
                knobs=model._knob_record(), log=model._log,
                speed_bounds=model._speed_bounds)
            _, dz_deep = ram_grid.compute_grid_lytaev(
                self.ENV, 10.0, max_range=1000.0, kind='ramgeo',
                zs=dz_free, knobs=model._knob_record(), log=model._log,
                speed_bounds=model._speed_bounds)
        assert dz_deep == dz_free

    def test_a_shallow_low_frequency_source_runs(self):
        src = Source(depths=5.0, frequencies=10.0)
        rcv = Receiver(depths=[5.0, 50.0], ranges=[500.0, 1000.0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = RAM(backend='ramgeo', verbose=False).run(
                self.ENV, src, rcv, run_mode=RunMode.COHERENT_TL)
        assert np.all(np.isfinite(np.asarray(field.dB, dtype=float)))


# ─── The section-gap bound survives the binary's single precision ─────────


class TestSectionGapBoundSurvivesSinglePrecision:
    """The binaries accumulate ``r = r + dr`` in default REAL and advance the
    bathymetry index once per step on ``r .ge. rb(ib+1)``
    (``ramgeo1.5.f:348``).
    A ``dr`` equal to the marker spacing lands a few float32 ulps below its
    marker on most steps, and the index trails by one segment for the whole
    march; the bound sits strictly inside the gap by more than that drift."""

    N_STEPS = 2000

    @staticmethod
    def _lag(dr, gap, n_steps):
        """Largest number of markers the binary's index trails the running
        range by, on a uniform marker grid at ``gap``, both read back as the
        deck spells them (``%.12g``) into REAL."""
        f32 = np.float32
        dr32 = f32(float(f"{dr:.12g}"))
        markers = np.array([f32(float(f"{k * gap:.12g}"))
                            for k in range(1, n_steps + 2)])
        r, ib, worst = f32(0.0), 0, 0
        for _ in range(n_steps):
            r = f32(r + dr32)
            if r >= markers[ib]:
                ib += 1
            passed = int(np.searchsorted(markers, r, side='right'))
            worst = max(worst, passed - ib)
        return worst

    @pytest.mark.parametrize('gap', [7.3, 13.7, 33.3])
    def test_the_index_never_trails_the_march(self, gap):
        segs = [{'range': k * gap} for k in range(4)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dr = ram_collins.constrain_dr_to_sections(
                500.0, segs, pinned=False, log=RAM(backend='ramgeo',
                                                   verbose=False)._log)
        assert dr < gap
        assert self._lag(dr, gap, self.N_STEPS) == 0

    @pytest.mark.parametrize('gap', [7.3, 13.7, 33.3])
    def test_the_gap_itself_would_trail(self, gap):
        # The other side of the threshold: at dr == gap the drift bites.
        assert self._lag(gap, gap, self.N_STEPS) > 0


# ─── The Collins deck carries the flat-earth transform ────────────────────


class TestCollinsDeckCarriesTheFlatEarthTransform:
    """``earth_curvature=True`` is the default on every backend. mpiramS applies
    ``peramx.f90:268-281`` inside the binary; the Collins decks get the same
    map from uacpy — ``eps = z/Re``, ``z' = z(1 + eps/2 + eps²/3)``,
    ``c' = c(1 + eps + eps²)`` — and their output depth axis is mapped back
    the way ``peramx.f90:444-449`` maps mpiramS's."""

    RE = 6378137.0

    @classmethod
    def _map(cls, z):
        eps = z / cls.RE
        return z * (1 + eps / 2 + eps * eps / 3), 1 + eps + eps * eps

    def test_water_rows_are_mapped(self):
        env = _halfspace_env(4000.0, [(0.0, 1500.0), (4000.0, 1520.0)])
        knobs = RAM(verbose=False, earth_curvature=True)._knob_record()
        rows = ram_collins.collins_deck_base(
            env, 'ramgeo', 4600.0, knobs=knobs)['segments'][0]['water_ssp']
        z_map, c_fac = self._map(4000.0)
        assert rows[0] == (0.0, 1500.0)
        assert rows[1] == pytest.approx((z_map, 1520.0 * c_fac), rel=1e-12)
        assert rows[1][0] - 4000.0 == pytest.approx(1.2548, abs=1e-3)

    def test_the_whole_deck_sits_in_one_frame(self):
        """The deck's floor, the water block's end, the sediment block's end
        and the absorbing ramp are all the geometric values under one map:
        the ramgeo block ends at ``map(zmax) − map(seafloor)`` and the ramp
        is at least as wide as its geometric width."""
        # The profile runs past zmax so the water block is cut there too.
        env = _halfspace_env(4000.0, [(0.0, 1500.0), (5000.0, 1525.0)])
        zmax = 4600.0
        mapped = RAM(verbose=False, earth_curvature=True)
        raw = RAM(verbose=False, earth_curvature=False)
        seg_m = ram_collins.collins_range_segments(
            env, 'ramgeo', zmax, 20.0, knobs=mapped._knob_record(),
            speed_bounds=mapped._speed_bounds)[0]
        seg_r = ram_collins.collins_range_segments(
            env, 'ramgeo', zmax, 20.0, knobs=raw._knob_record(),
            speed_bounds=raw._speed_bounds)[0]
        z_floor = ram_domain.deck_depth(
            zmax, knobs=mapped._knob_record()) - ram_domain.deck_depth(
                4000.0, knobs=mapped._knob_record())
        assert seg_m['water_ssp'][-1][0] == pytest.approx(
            ram_domain.deck_depth(zmax, knobs=mapped._knob_record()),
            rel=1e-12)
        assert seg_m['bottom_c'][-1][0] == pytest.approx(z_floor, rel=1e-12)
        assert seg_m['bottom_attn'][-1][0] == pytest.approx(z_floor, rel=1e-12)
        assert seg_r['bottom_c'][-1][0] == pytest.approx(600.0)
        assert z_floor - 600.0 == pytest.approx(0.4045, abs=1e-3)
        width_m = seg_m['bottom_attn'][-1][0] - seg_m['bottom_attn'][-2][0]
        width_r = seg_r['bottom_attn'][-1][0] - seg_r['bottom_attn'][-2][0]
        assert width_m >= width_r
        assert width_m == pytest.approx(width_r, rel=2e-3)

    def test_flat_earth_false_writes_the_raw_profile(self):
        env = _halfspace_env(4000.0, [(0.0, 1500.0), (4000.0, 1520.0)])
        knobs = RAM(verbose=False, earth_curvature=False)._knob_record()
        rows = ram_collins.collins_deck_base(
            env, 'ramgeo', 4600.0, knobs=knobs)['segments'][0]['water_ssp']
        assert rows == [(0.0, 1500.0), (4000.0, 1520.0)]

    def test_depths_round_trip_through_the_map(self):
        model = RAM(verbose=False, earth_curvature=True)
        z = np.array([0.0, 100.0, 4000.0])
        back = ram_domain.deck_depth_inverse(
            ram_domain.deck_depth(z, knobs=model._knob_record()),
            knobs=model._knob_record())
        np.testing.assert_allclose(back, z, rtol=0, atol=1e-9)
        assert ram_domain.deck_depth(
            4000.0,
            knobs=model._knob_record()) == pytest.approx(self._map(4000.0)[0])

    def test_flat_earth_is_a_setting_of_every_backend(self):
        assert 'earth_curvature' not in [n for n, _ in RAM._MPIRAMS_ONLY_SETTINGS]
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            RAM(backend='ramgeo', verbose=False,
                earth_curvature=False)._warn_on_mpirams_only_overrides('ramgeo')

    def test_the_output_depth_axis_is_mapped_back(self):
        """The binary's grid is ``k·dz`` in the deck's frame; the axis handed
        to the receiver interpolation is geometric, so its image under the
        map is the grid."""
        env = _halfspace_env(4000.0, [(0.0, 1500.0), (4000.0, 1520.0)])
        src = Source(depths=100.0, frequencies=20.0)
        rcv = Receiver(depths=[100.0, 3990.0], ranges=[2000.0, 4000.0])
        model = RAM(backend='ramgeo', verbose=False, dz=5.0, dr=50.0, earth_curvature=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = _collins_launch_grid(model, env, src, rcv)
        depths = np.asarray(raw['depths'], dtype=float)
        grid = ram_domain.deck_depth(depths, knobs=model._knob_record())
        np.testing.assert_allclose(grid / 5.0, np.round(grid / 5.0),
                                   rtol=0, atol=1e-6)
        deep = depths[depths > 3900.0]
        assert deep.size
        assert np.all(np.abs(deep / 5.0 - np.round(deep / 5.0)) > 0.1)


# ─── A non-vacuum sea surface collapses to pressure release, loudly ───────


class TestNonVacuumSurfaceCollapsesToPressureRelease:
    """No RAM deck carries a surface record and every binary holds the top
    row at zero pressure, so a rigid or ice surface is run as vacuum. The
    caller is told, once, which surface kind was dropped."""

    @staticmethod
    def _env(surface):
        return Environment(
            bathymetry=100.0, ssp=1500.0, surface=surface,
            bottom=Bottom.from_halfspace(BoundaryProperties(
                sound_speed=1700.0, density=1.8, attenuation=0.5)))

    RIGID = BoundaryProperties(acoustic_type='rigid')
    FLUID_ICE = BoundaryProperties(acoustic_type='half-space',
                                   sound_speed=3500.0, density=0.9,
                                   attenuation=0.4)
    ELASTIC_ICE = BoundaryProperties(acoustic_type='half-space',
                                     sound_speed=3500.0, density=0.9,
                                     attenuation=0.4, shear_speed=1800.0,
                                     shear_attenuation=1.0)

    @pytest.mark.parametrize('surface, kind', [
        (RIGID, "'rigid'"), (FLUID_ICE, "'half-space'"),
        (ELASTIC_ICE, "'half-space'"),
    ])
    def test_a_non_vacuum_surface_warns_and_becomes_vacuum(self, surface,
                                                           kind):
        with pytest.warns(UserWarning,
                          match='pressure-release surface') as rec:
            out = ram_domain.collapse_surface_to_pressure_release(
                self._env(surface))
        texts = [str(w.message) for w in rec
                 if 'pressure-release surface' in str(w.message)]
        assert len(texts) == 1
        assert kind in texts[0] and 'modelled as vacuum' in texts[0]
        assert out.surface.acoustic_type == 'vacuum'
        assert out.surface.shear_speed == 0.0

    def test_an_elastic_surface_names_the_shear_too(self):
        with pytest.warns(UserWarning, match='surface shear is not supported'):
            ram_domain.collapse_surface_to_pressure_release(
                self._env(self.ELASTIC_ICE))

    def test_a_vacuum_surface_is_left_alone_without_a_warning(self):
        env = self._env(None)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = ram_domain.collapse_surface_to_pressure_release(env)
        assert out is env

    def test_the_collapsed_deck_is_the_vacuum_deck(self):
        model = RAM(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            collapsed = ram_domain.collapse_surface_to_pressure_release(
                self._env(self.RIGID))
        assert _exact(ram_collins.collins_range_segments(
            collapsed, 'ramgeo', 400.0, 100.0, knobs=model._knob_record(),
            speed_bounds=model._speed_bounds)) == _exact(
            ram_collins.collins_range_segments(
                self._env(None), 'ramgeo', 400.0, 100.0,
                knobs=model._knob_record(), speed_bounds=model._speed_bounds))


class TestTheCollinsStrideCapsTheAutomaticRangeStep:
    """The Collins binaries write the field every step and nowhere else, so
    an automatic ``dr`` is also the output stride the receiver modulus is
    interpolated across; it is capped at ``COLLINS_SAMPLES_PER_BEAT``
    samples per modal-beat period (``2π/Δk``). mpiramS marches onto every
    receiver range and carries no such cap; a pinned ``dr`` is the caller's."""

    @staticmethod
    def _sand():
        return make_pekeris(sound_speed=1600.0, density=1.5)

    def test_ramgeo_is_capped_at_the_beat_fraction(self, monkeypatch):
        from uacpy.models.ram._domain import COLLINS_SAMPLES_PER_BEAT
        m = _stub_grid(monkeypatch, RAM(backend='ramgeo', verbose=False),
                       dr=100.0, dz=1.0)
        env = self._sand()
        beat = 2 * np.pi / ram_domain.modal_beat_wavenumber(
            env, 100.0, speed_bounds=m._speed_bounds)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dr, _ = ram_grid.compute_grid_lytaev(env, 100.0, max_range=5000.0,
                                           kind='ramgeo',
                                           knobs=m._knob_record(), log=m._log,
                                           speed_bounds=m._speed_bounds)
        assert beat / COLLINS_SAMPLES_PER_BEAT < 100.0, "cap must bind"
        assert dr == pytest.approx(beat / COLLINS_SAMPLES_PER_BEAT)

    def test_mpirams_keeps_the_optimisers_step(self, monkeypatch):
        m = _stub_grid(monkeypatch, RAM(backend='mpirams', verbose=False),
                       dr=100.0, dz=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dr, _ = ram_grid.compute_grid_lytaev(self._sand(), 100.0,
                                           max_range=5000.0, kind='mpirams',
                                           knobs=m._knob_record(), log=m._log,
                                           speed_bounds=m._speed_bounds)
        assert dr == pytest.approx(100.0)

    def test_a_step_already_under_the_cap_is_untouched(self, monkeypatch):
        m = _stub_grid(monkeypatch, RAM(backend='ramgeo', verbose=False),
                       dr=10.0, dz=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dr, _ = ram_grid.compute_grid_lytaev(self._sand(), 100.0,
                                           max_range=5000.0, kind='ramgeo',
                                           knobs=m._knob_record(), log=m._log,
                                           speed_bounds=m._speed_bounds)
        assert dr == pytest.approx(10.0)

    def test_a_pinned_dr_is_the_callers(self, monkeypatch):
        m = _stub_grid(monkeypatch, RAM(backend='ramgeo', dr=100.0,
                                        verbose=False),
                       dr=100.0, dz=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dr, _dz, _zmax = ram_collins.resolve_collins_grid(
                self._sand(), 100.0, 'ramgeo', 5000.0, None, None, None,
                knobs=m._knob_record(), log=m._log,
                speed_bounds=m._speed_bounds)
        assert dr == pytest.approx(100.0)

    def test_six_samples_per_beat_is_the_measured_bound(self):
        """The constant is the measured rung, not a free knob: on ramgeo at
        200 Hz over the sand channel (beat 120 m) the receiver interpolation
        between writes is 3.7 % (0.3 dB) off a dr = 1 m march at a beat/3
        stride and 1.0 % (0.1 dB) at beat/6, the first rung under the λ/16
        floor's own error. Three would pass every other test in this class
        and hand back the beat/3 error."""
        from uacpy.models.ram._domain import COLLINS_SAMPLES_PER_BEAT
        from uacpy.models.ram.grid import collins_output_spacing
        assert COLLINS_SAMPLES_PER_BEAT == 6.0
        assert collins_output_spacing(2 * np.pi / 120.0) == pytest.approx(20.0)

    def test_the_dr_cap_and_the_ndr_cap_are_one_spacing(self):
        """The stride marched is dr·ndr: with dr at the cap, ndr stays 1
        however long the run (a 50 km run once logged six samples per beat
        while its ndr of 2 made the stride a third of the beat), and at half
        the cap ndr is exactly 2 — the same spacing bounds both."""
        from uacpy.models.ram.grid import collins_output_spacing
        m = RAM(backend='ramgeo', verbose=False)
        beat_k = ram_domain.modal_beat_wavenumber(self._sand(), 200.0,
                                                  speed_bounds=m._speed_bounds)
        cap = collins_output_spacing(beat_k)
        ranges = [1000.0, 50_000.0]
        assert ram_collins.collins_output_stride(cap, 50_000.0, ranges,
                                          beat_wavenumber=beat_k)[0] == 1
        assert ram_collins.collins_output_stride(cap / 2, 50_000.0, ranges,
                                          beat_wavenumber=beat_k)[0] == 2


class TestTheSeafloorSitsAQuarterCellBelowItsWaterNode:
    """On the fluid backends the automatic ``dz`` puts the shallowest
    seafloor ``SEAFLOOR_CELL_OFFSET`` of a cell below its last water node,
    ``h/dz = n + offset``; rams0.5 keeps it on the node. Every backend's
    ``iz`` truncation then lands on the same node under the deck's 12-digit
    spelling — the placement sits a quarter cell from the cliff."""

    @staticmethod
    def _env(depth=100.0):
        return Environment(
            bathymetry=depth, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))

    def _auto_dz(self, kind, depth=100.0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = RAM(backend=kind, verbose=False)
            _, dz = ram_grid.compute_grid_lytaev(
                self._env(depth), 200.0, max_range=5000.0, kind=kind,
                knobs=model._knob_record(), log=model._log,
                speed_bounds=model._speed_bounds)
        return dz

    def test_the_offset_is_the_measured_quarter_cell(self):
        """The value is measured, not free: on the grids the automatic path
        marches, the node itself is never the best placement and a quarter
        cell wins or ties in 8 of 10 cases (sand 200 Hz far field 1.02 dB
        on the node, 0.32 a quarter cell below, 0.72 mid-cell)."""
        from uacpy.models.ram.grid import SEAFLOOR_CELL_OFFSET
        assert SEAFLOOR_CELL_OFFSET == 0.25

    @pytest.mark.parametrize('kind', ['mpirams', 'ramgeo', 'ramsurf'])
    @pytest.mark.parametrize('depth', [100.0, 87.3])
    def test_a_fluid_backend_places_the_seafloor_a_quarter_cell_down(
            self, kind, depth):
        from uacpy.models.ram.grid import SEAFLOOR_CELL_OFFSET
        dz = self._auto_dz(kind, depth)
        ratio = depth / dz
        assert ratio - np.floor(ratio) == pytest.approx(SEAFLOOR_CELL_OFFSET,
                                                        abs=1e-9)
        # The truncation lands on the same node under either deck spelling.
        n = int(np.floor(ratio))
        for spelled in (float(f"{dz:.12g}"), float(repr(dz))):
            assert int(1.0 + depth / spelled) == n + 1

    def test_rams_keeps_the_seafloor_on_the_node(self):
        column = _env_range_dependent_elastic().bottom.columns[0]
        env = Environment(bathymetry=100.0, ssp=1500.0, bottom=column)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = RAM(backend='rams', verbose=False)
            _, dz = ram_grid.compute_grid_lytaev(
                env, 50.0, max_range=5000.0, kind='rams',
                knobs=model._knob_record(), log=model._log,
                speed_bounds=model._speed_bounds)
        ratio = 100.0 / dz
        assert ratio == pytest.approx(round(ratio), abs=1e-6)

    def test_the_alignment_helper_keeps_the_offset_on_both_sides(self):
        from uacpy.models.ram.grid import SEAFLOOR_CELL_OFFSET
        model = RAM(verbose=False)
        env = self._env()
        tighter = ram_grid.align_dz_with_seafloor(env, 0.4, kind='ramgeo',
                                                  knobs=model._knob_record())
        coarser = ram_grid.align_dz_with_seafloor(env, 0.4, kind='ramgeo',
                                                coarsen=True,
                                                knobs=model._knob_record())
        assert tighter <= 0.4 < coarser
        for dz in (tighter, coarser):
            ratio = 100.0 / dz
            assert ratio - np.floor(ratio) == pytest.approx(
                SEAFLOOR_CELL_OFFSET, abs=1e-9)
        # One layer apart: the two bracket the raw value.
        assert 100.0 / tighter - 100.0 / coarser == pytest.approx(1.0)

    def test_a_placed_dz_is_left_where_it_is(self):
        model = RAM(verbose=False)
        placed = ram_grid.dz_for_water_layers(100.0, 250, 'mpirams')
        aligned = ram_grid.align_dz_with_seafloor(self._env(), placed,
                                                kind='mpirams',
                                                knobs=model._knob_record())
        assert aligned == pytest.approx(placed, rel=1e-9)


class TestTheTrappedModeWarning:
    """When the seabed's critical angle is wider than the aperture, the
    steepest scored component is a trapped mode that reaches the receivers;
    a score at or above ``TRAPPED_MODE_SCORE_LIMIT`` means its phase is
    lost. The automatic grid refines ``dz`` from the λ/16 floor until the
    mode passes, within ``MAX_DEPTH_POINTS``; the grid MARCHED — pinned, or
    automatic and cut short by the budget — warns at the default
    ``accuracy`` when it still cannot carry the mode, naming a ``dz`` that
    would. A grid that scores under the limit is silent.

    ``_model`` stubs the search at rock's own dr = 19 m so the dz side is
    what each test exercises."""

    @staticmethod
    def _model(monkeypatch, **pinned):
        m = RAM(backend='mpirams', verbose=False, **pinned)
        monkeypatch.setattr(
            'uacpy.models.ram.grid.optimize_grid_relaxing',
            lambda **kw: ({'dr': 19.0, 'dz': 0.01}, kw['eps0'], kw['theta0']))
        logged = []
        m._log = lambda msg, level='info', _l=logged: _l.append(msg)
        return m, logged

    @staticmethod
    def _env(c_b):
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=c_b, density=2.2,
                                      attenuation=0.2))

    def _grid(self, monkeypatch, c_b, **pinned):
        m, logged = self._model(monkeypatch, **pinned)
        with recorded_warnings() as caught:
            dr, dz = ram_mpirams.resolve_mpirams_grid(
                self._env(c_b), 200.0, 5000.0, knobs=m._knob_record(),
                log=m._log, speed_bounds=m._speed_bounds)
        texts = [str(w.message) for w in caught
                 if 'traps modes' in str(w.message)]
        return m, dr, dz, texts, logged

    def _score(self, m, c_b, dr, dz):
        from uacpy.models.pe_grid import optimize_grid
        c0 = ram_domain.resolve_c0(self._env(c_b), knobs=m._knob_record(),
                                   speed_bounds=m._speed_bounds)
        return optimize_grid(
            grid=(dr, dz), frequency=200.0, c_min=1500.0, c_max=c0, x_max=5000.0,
            c0=c0, angle_max=30.0, p=int(m.n_pade), alpha=0.0,
            c_min_all=1500.0, c_max_all=c_b)

    def test_rock_gets_dz_refined_below_the_floor_and_no_warning(self,
                                                                 monkeypatch):
        """The λ/16 floor is where the depth search starts: on the floored
        grid the 51° trapped mode scores 16, so dz is refined to the
        coarsest value that scores under the limit, and nothing warns."""
        from uacpy.models.ram.grid import TRAPPED_MODE_SCORE_LIMIT
        # The limit is where the per-step score stops ranking grids (it
        # saturates at 2 per step): a lower one doubles the depth points for
        # the ≤ 0.5 dB the λ/250 placements leave, a higher one marches a
        # lost phase (rock at 200 Hz: score 4 reads 6.3 dB, score 1 reads
        # 1.4 with the seafloor placed).
        assert TRAPPED_MODE_SCORE_LIMIT == 1.0
        m, dr, dz, texts, logged = self._grid(monkeypatch, 2400.0)
        floor = 1500.0 / (16 * 200.0)
        assert dz < floor
        assert texts == []
        assert [ln for ln in logged if 'refined dz from' in ln]
        passing = self._score(m, 2400.0, dr, dz)
        assert passing['trapped_end_binds']
        assert passing['predicted_error'] < TRAPPED_MODE_SCORE_LIMIT
        # Coarsest such dz: two per cent coarser already fails.
        failing = self._score(m, 2400.0, dr, dz * 1.02)
        assert failing['predicted_error'] >= TRAPPED_MODE_SCORE_LIMIT

    def test_the_depth_budget_stops_the_refinement_and_the_warning_names_it(
            self):
        """Both sides of ``MAX_DEPTH_POINTS``: the dz hard rock needs at
        200 Hz is ~λ/162 (4.6 cm), which 4 km of water holds under the
        100 000-point budget and 6 km does not. Past it the automatic dz stops at the budget and
        the marched-grid warning names the need and the budget."""
        from uacpy.models.ram._domain import MAX_DEPTH_POINTS
        from uacpy.models.ram.grid import SEAFLOOR_CELL_OFFSET

        def grid(h):
            # The real search: hard rock needs its own dr (2.9 m), not the
            # stub's 19 m, for any dz to carry the 74° mode.
            m = RAM(backend='mpirams', verbose=False)
            logged = []
            m._log = lambda msg, level='info', _l=logged: _l.append(msg)
            env = Environment(
                bathymetry=h, ssp=1500.0,
                bottom=BoundaryProperties(acoustic_type='half-space',
                                          sound_speed=5500.0, density=2.6,
                                          attenuation=0.1))
            with recorded_warnings() as caught:
                dr, dz = ram_mpirams.resolve_mpirams_grid(
                    env, 200.0, 5000.0, knobs=m._knob_record(), log=m._log,
                    speed_bounds=m._speed_bounds)
            texts = [str(w.message) for w in caught
                     if 'traps modes' in str(w.message)]
            return dr, dz, texts, logged

        _dr, dz, texts, _ = grid(4000.0)
        assert 4000.0 / dz < MAX_DEPTH_POINTS
        assert texts == []
        _dr, dz, texts, logged = grid(6000.0)
        assert 6000.0 / dz == pytest.approx(
            MAX_DEPTH_POINTS + SEAFLOOR_CELL_OFFSET, abs=1e-6)
        assert len(texts) == 1
        assert f'{MAX_DEPTH_POINTS}-point budget' in texts[0]
        assert 'MAX_DEPTH_POINTS' in texts[0] and 'dz <=' in texts[0]
        assert [ln for ln in logged if 'past the' in ln and 'budget' in ln]

    def test_sand_does_not_warn(self, monkeypatch):
        """Sand's 20° critical angle sits inside the 30° aperture: the
        steepest scored component is the aperture edge, not a trapped
        mode, and the floored grid is the documented about-a-decibel case."""
        _m, _dr, _dz, texts, _ = self._grid(monkeypatch, 1600.0)
        assert texts == []

    def test_the_named_dz_scores_under_the_limit_at_the_marched_dr(
            self, monkeypatch):
        from uacpy.models.ram.grid import TRAPPED_MODE_SCORE_LIMIT
        m, dr, _dz, _texts, logged = self._grid(monkeypatch, 2400.0, dr=19.0,
                                                dz=1.0)
        advice = [ln for ln in logged if 'pin dz to march it' in ln][0]
        dz_need = float(re.search(r'dz <= ([0-9.e+-]+) m', advice).group(1))
        score = self._score(m, 2400.0, dr, dz_need)
        assert score['predicted_error'] < TRAPPED_MODE_SCORE_LIMIT
        assert score['trapped_end_binds']
        assert f'dz <= {dz_need:.3g} m' in _texts[0]

    def test_a_pinned_dz_at_the_advised_value_is_silent(self, monkeypatch):
        """The check scores the grid marched: pin dz to the value the log
        named (with the same pinned dr) and no warning follows."""
        _m, dr, _dz, _texts, logged = self._grid(monkeypatch, 2400.0, dr=19.0,
                                                 dz=1.0)
        advice = [ln for ln in logged if 'pin dz to march it' in ln][0]
        dz_need = float(re.search(r'dz <= ([0-9.e+-]+) m', advice).group(1))
        _m, dr_p, dz_p, texts, logged = self._grid(monkeypatch, 2400.0,
                                                   dr=19.0, dz=dz_need)
        assert dz_p == pytest.approx(dz_need) and dr_p == pytest.approx(dr)
        assert texts == []
        assert not [ln for ln in logged if 'pin dz to march it' in ln]

    def test_a_pinned_coarse_grid_is_warned_about_as_itself(self, monkeypatch):
        """Both steps pinned: the Lytaev chooser never runs, and the warning
        describes the pinned pair, not an automatic grid."""
        _m, dr, dz, texts, logged = self._grid(monkeypatch, 2400.0, dr=19.0,
                                               dz=1.0)
        assert (dr, dz) == (19.0, 1.0)
        assert not [ln for ln in logged if 'Lytaev grid' in ln]
        assert len(texts) == 1
        assert 'dr=19 m, dz=1 m' in texts[0] and '51°' in texts[0]

    def test_the_grid_line_reports_both_bands(self, monkeypatch):
        _m, _dr, _dz, _texts, logged = self._grid(monkeypatch, 2400.0)
        line = [ln for ln in logged if 'Lytaev grid' in ln][0]
        assert 'predicted error' in line
        assert 'water band' in line and 'critical angle 51°' in line
        assert 'stability-band growth' in line


# ── Water-column volume attenuation on every RAM backend ─────────────────────
# The Collins decks carry ``alpha(z)`` as one more profile block per section in
# dB per local wavelength, announced by a fifth number on row 5; mpiramS reads
# a depth x bin table named on the last line of ``in.pe``. The binaries apply
# it to the water wavenumber as ``k(1 + i*eta*beta)`` — the seabed's own form
# (``third_party/MODIFICATIONS.md``, *water-column attenuation*).

def _segment(with_water, rho=1.5, shear=False):
    seg = dict(
        range=0.0,
        water_ssp=[(0.0, 1500.0), (100.0, 1500.0)],
        bottom_c=[(0.0, 1700.0), (200.0, 1700.0)],
        bottom_rho=[(0.0, rho), (200.0, rho)],
        bottom_attn=[(0.0, 0.5), (200.0, 0.5)],
    )
    if shear:
        seg['bottom_cs'] = [(0.0, 300.0), (200.0, 300.0)]
        seg['bottom_attns'] = [(0.0, 1.0), (200.0, 1.0)]
    if with_water:
        seg['water_attn'] = [(0.0, 0.001), (200.0, 0.001)]
    return seg


def _write(path, kind, segments, **kw):
    args = dict(kind=kind, frequency=400.0, zs=30.0, zr_line=30.0, rmax_march=3000.0,
                dr=5.0, ndr=1, zmax=200.0, dz=0.25, depth_decimation=1, zmplt=150.0,
                c0=1500.0, n_pade=4,
                bathymetry=[(0.0, 100.0), (3000.0, 100.0)],
                range_segments=segments)
    if kind == 'ramsurf':
        args['surface'] = [(0.0, 0.0), (3000.0, 0.0)]
    if kind == 'rams':
        args.update(rams_rotation=1, rams_rotation_angle=45.0)
    args.update(kw)
    write_ramin(str(path), **args)
    return path.read_text().splitlines()


class TestRamInWaterBlock:

    def test_block_in_every_section_sets_row5_and_adds_one_terminator_each(
            self, tmp_path):
        segs = [_segment(True), dict(_segment(True), range=1500.0)]
        lines = _write(tmp_path / 'ramgeo.in', 'ramgeo', segs)
        assert lines[4] == '1500 4 1 0 1'
        # bathymetry + 2 sections x (cw, cb, rho, attn, attw)
        assert sum(1 for ln in lines if ln.startswith('-1')) == 11
        # the water block is the LAST block of a section: its value line sits
        # right before the range line that opens the next section.
        i = lines.index('1500')
        assert lines[i - 1] == '-1 -1' and lines[i - 2] == '200 0.001'

    def test_without_the_block_row5_keeps_four_numbers(self, tmp_path):
        lines = _write(tmp_path / 'ramgeo.in', 'ramgeo', [_segment(False)])
        assert lines[4] == '1500 4 1 0'
        assert sum(1 for ln in lines if ln.startswith('-1')) == 5

    def test_rams_row5_carries_the_fifth_number_after_theta(self, tmp_path):
        lines = _write(tmp_path / 'rams.in', 'rams',
                       [_segment(True, shear=True)])
        assert lines[4] == '1500 4 1 45 1'
        assert sum(1 for ln in lines if ln.startswith('-1')) == 8

    def test_a_block_in_some_sections_only_is_refused(self, tmp_path):
        segs = [_segment(True), dict(_segment(False), range=1500.0)]
        with pytest.raises(ConfigurationError, match='every range segment'):
            _write(tmp_path / 'ramgeo.in', 'ramgeo', segs)


class TestMpiramsWaterTable:

    def _inpe(self, path, **kw):
        write_inpe(path, fc=400.0, q_factor=1e6, record_duration=1.0, zsrc=30.0, dz=0.25,
                   dr=5.0, n_pade=4, n_stability=1, stability_range_m=3000.0, depth_decimation=1,
                   ssp_filename='ssp.dat', earth_curvature=0, horizontal_interpolation=0, bathymetry_from_file=1,
                   bth_filename='bth.dat', sedlayer=50.0, n_sediment_points=4,
                   cs=np.zeros(4), rho=np.full(4, 1.5), attn=np.full(4, 0.5),
                   c0=1500.0, **kw)
        return path.read_text().splitlines()

    def test_table_name_is_the_decks_last_line(self, tmp_path):
        lines = self._inpe(tmp_path / 'in.pe', water_attn_filename='w.dat')
        assert lines[-1] == 'w.dat'
        assert lines[-2].split() == ['0.5'] * 4

    def test_without_a_table_the_deck_ends_with_the_attenuation_row(
            self, tmp_path):
        lines = self._inpe(tmp_path / 'in.pe')
        assert lines[-1].split() == ['0.5'] * 4

    def test_table_layout_is_header_frequencies_then_one_row_per_depth(
            self, tmp_path):
        out = tmp_path / 'w.dat'
        table = np.array([[1e-4, 2e-4, 3e-4], [4e-4, 5e-4, 6e-4]])
        write_water_attenuation_file(out, [0.0, 150.0], [399.0, 400.0, 401.0],
                                     table)
        lines = out.read_text().splitlines()
        assert lines[0] == '2 3'
        assert [float(v) for v in lines[1].split()] == [399.0, 400.0, 401.0]
        assert [float(v) for v in lines[2].split()] == [0.0, 1e-4, 2e-4, 3e-4]
        assert [float(v) for v in lines[3].split()] == [150.0, 4e-4, 5e-4, 6e-4]

    def test_non_monotone_depths_and_negative_values_are_refused(
            self, tmp_path):
        with pytest.raises(ConfigurationError, match='increase strictly'):
            write_water_attenuation_file(tmp_path / 'w.dat', [0.0, 0.0],
                                         [400.0], np.zeros((2, 1)))
        with pytest.raises(ConfigurationError, match='non-negative'):
            write_water_attenuation_file(tmp_path / 'w.dat', [0.0, 1.0],
                                         [400.0], -np.ones((2, 1)))


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
class TestWrapperBlock:

    def _ram(self):
        from uacpy.models import RAM
        return RAM(verbose=False, earth_curvature=False)

    def test_thorp_block_is_dB_per_local_wavelength(self):
        env = Environment(name='grad', bathymetry=100.0,
                          ssp=[(0.0, 1500.0), (100.0, 1480.0)],
                          absorption=Thorp())
        block = ram_domain.water_attenuation_block(
            env, 10000.0, env.ssp.to_pairs(), 150.0)
        alpha_m = float(absorption_thorp(10000.0)) / 1000.0
        by_depth = dict(block)
        assert by_depth[0.0] == pytest.approx(alpha_m * 1500.0 / 10000.0)
        assert by_depth[100.0] == pytest.approx(alpha_m * 1480.0 / 10000.0)
        # below the profile's last sample the speed holds, so the block does
        assert by_depth[150.0] == pytest.approx(by_depth[100.0])

    def test_a_constant_block_carries_the_dB_per_wavelength_value_exactly(
            self, tmp_path):
        """A ConstantAbsorption is dB per local wavelength, the unit the AT
        decks carry verbatim; on a 1480 m/s layer a detour through dB/m at
        1500 m/s would write 1.3 % less. Both the Collins block and the
        mpiramS table carry the value unchanged at every depth."""
        env = Environment(name='grad', bathymetry=100.0,
                          ssp=[(0.0, 1500.0), (100.0, 1480.0)],
                          absorption=ConstantAbsorption(0.2))
        block = ram_domain.water_attenuation_block(
            env, 1000.0, env.ssp.to_pairs(), 150.0)
        assert [a for _, a in block] == [0.2] * len(block)
        name = ram_mpirams.write_mpirams_water_attenuation(
            env, tmp_path, 100.0, 1e6, 1.0, 150.0,
            knobs=self._ram()._knob_record())
        rows = (tmp_path / name).read_text().splitlines()[2:]
        table = np.array([[float(v) for v in row.split()] for row in rows])
        np.testing.assert_allclose(table[:, 1], 0.2, rtol=1e-6)

    def test_francois_garrison_is_evaluated_per_depth(self):
        fg = FrancoisGarrison(temperature=10.0, salinity=35.0, pH=8.0)
        env = Environment(name='fg', bathymetry=1000.0, ssp=1500.0,
                          absorption=fg)
        block = ram_domain.water_attenuation_block(
            env, 10000.0, env.ssp.to_pairs(), 1000.0)
        z = np.array([d for d, _ in block])
        a = np.array([v for _, v in block])
        expected = np.atleast_1d(fg.alpha_dB_per_m(10000.0, z)) * 1500.0 / 1e4
        np.testing.assert_allclose(a, expected, rtol=1e-12)
        assert a[-1] < a[0], "pressure lowers FG absorption with depth"

    def test_depths_thin_to_one_per_cell_and_stay_inside_the_domain(self):
        env = Environment(name='t', bathymetry=100.0,
                          ssp=[(0.0, 1500.0), (3.0, 1499.0), (100.0, 1480.0)],
                          absorption=Thorp())
        z = ram_domain.water_attenuation_depths(env, 150.0, dz=10.0)
        assert z[0] == 0.0 and z[-1] <= 150.0
        assert np.all(np.diff(z) >= 10.0)
        z_fine = ram_domain.water_attenuation_depths(env, 150.0)
        assert 3.0 in z_fine and 150.0 in z_fine

    def test_mpirams_table_has_one_column_per_sweep_bin(self, tmp_path):
        env = Environment(name='t', bathymetry=100.0, ssp=1500.0,
                          absorption=Thorp())
        m = self._ram()
        name = ram_mpirams.write_mpirams_water_attenuation(env, tmp_path,
                                                           100.0, 10.0,
                                                  1.0, 150.0,
                                                  knobs=m._knob_record())
        lines = (tmp_path / name).read_text().splitlines()
        frq = ram_band.broadband_frequencies(100.0, 10.0, 1.0)
        nz, nf = (int(v) for v in lines[0].split())
        assert nf == frq.size == 21
        np.testing.assert_allclose([float(v) for v in lines[1].split()], frq)
        rows = np.array([[float(v) for v in ln.split()] for ln in lines[2:]])
        assert rows.shape == (nz, nf + 1)
        j = 5
        expected = (float(absorption_thorp(frq[j])) / 1000.0 * 1500.0 / frq[j])
        np.testing.assert_allclose(rows[:, j + 1], expected, rtol=1e-9)


def _tl_line(path):
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    return np.array([[float(r[0]), float(r[1])] for r in rows])


@pytest.mark.requires_binary
@pytest.mark.slow
class TestCollinsBinariesApplyTheBlock:
    """Deck + binary, no wrapper: the patched engines read the fifth block
    and lose the plane-wave ``beta * r / lambda`` over the water path; a
    block of zeros changes nothing."""

    BETA = 0.02  # dB/wavelength; 800 wavelengths at 3 km and 400 Hz = 16 dB

    def _run(self, tmp_path, kind, water):
        from uacpy.models import RAM
        ram = RAM(verbose=False)
        d = tmp_path / f'{kind}_{water}'
        d.mkdir()
        seg = _segment(False, shear=(kind == 'rams'))
        if water is not None:
            seg['water_attn'] = [(0.0, water), (200.0, water)]
        _write(d / ram_collins.collins_in_name(kind), kind, [seg])
        subprocess.run([str(ram._collins_binary(kind))], cwd=d, check=True,
                       capture_output=True, timeout=600)
        return d

    @pytest.mark.parametrize('kind', ['ramgeo', 'rams', 'ramsurf'])
    def test_extra_loss_is_the_plane_wave_value(self, tmp_path, kind):
        plain = _tl_line(self._run(tmp_path, kind, None) / 'tl.line')
        lossy = _tl_line(self._run(tmp_path, kind, self.BETA) / 'tl.line')
        r = plain[:, 0]
        extra = lossy[:, 1] - plain[:, 1]
        expected = self.BETA * r / (1500.0 / 400.0)
        tail = r > 2000.0
        ratio = np.median(extra[tail] / expected[tail])
        assert 0.85 < ratio < 1.3, (
            f"{kind}: extra loss is {ratio:.2f} x the plane-wave value")

    @pytest.mark.parametrize('kind', ['ramgeo', 'rams', 'ramsurf'])
    def test_a_block_of_zeros_reproduces_the_lossless_field(self, tmp_path,
                                                            kind):
        plain = self._run(tmp_path, kind, None)
        zero = self._run(tmp_path, kind, 0.0)
        if kind == 'ramsurf':
            # the complex square rounds differently from the real one
            d = np.abs(_tl_line(plain / 'tl.line')[:, 1]
                       - _tl_line(zero / 'tl.line')[:, 1])
            assert d.max() < 1e-6
        else:
            assert (plain / 'tl.grid').read_bytes() == \
                (zero / 'tl.grid').read_bytes()
            assert (plain / 'pcomplex.bin').read_bytes() == \
                (zero / 'pcomplex.bin').read_bytes()


def _pekeris(backend, absorption):
    kw = {}
    if backend == 'rams':
        bottom = BoundaryProperties(sound_speed=1700.0, shear_speed=300.0,
                                    density=1.8, attenuation=0.5,
                                    shear_attenuation=1.0)
    else:
        bottom = BoundaryProperties(sound_speed=1700.0, density=1.8,
                                    attenuation=0.5)
    if backend == 'ramsurf':
        kw['altimetry'] = [(0.0, 0.0), (3000.0, 0.0)]
    return Environment(name=f'pekeris-{backend}', bathymetry=100.0,
                       ssp=1500.0, bottom=bottom, absorption=absorption, **kw)


@pytest.mark.requires_binary
@pytest.mark.slow
class TestEveryBackendAppliesEnvAbsorption:
    """The wrapper route: the same ``Environment`` with and without a
    constant 0.02 dB/wavelength loses the plane-wave value on every backend,
    and mpiramS's broadband table lowers every bin."""

    BETA = 0.02
    RANGES = [1000.0, 2000.0, 3000.0]

    def _tl(self, backend, absorption):
        from uacpy.models import RAM
        ram = RAM(backend=backend, verbose=False)
        field = ram.run(_pekeris(backend, absorption),
                        Source(depths=30.0, frequencies=400.0),
                        Receiver(depths=[30.0], ranges=self.RANGES),
                        run_mode=RunMode.COHERENT_TL)
        return np.asarray(field.tl, dtype=float).reshape(-1)

    @pytest.mark.parametrize('backend', ['mpirams', 'ramgeo', 'rams', 'ramsurf'])
    def test_constant_absorption_costs_beta_r_over_lambda(self, backend):
        extra = (self._tl(backend, ConstantAbsorption(self.BETA))
                 - self._tl(backend, None))
        expected = self.BETA * np.asarray(self.RANGES) / (1500.0 / 400.0)
        ratio = extra / expected
        assert np.all((0.8 < ratio) & (ratio < 1.35)), (
            f"{backend}: extra loss / plane-wave = {ratio}")

    def test_mpirams_broadband_table_lowers_every_bin(self):
        from uacpy.models import RAM
        freqs = np.array([380.0, 390.0, 400.0, 410.0, 420.0])
        src = Source(depths=30.0, frequencies=freqs)
        rcv = Receiver(depths=[30.0], ranges=[3000.0])
        H = {}
        for label, absorption in (('plain', None),
                                  ('lossy', ConstantAbsorption(self.BETA))):
            field = RAM(backend='mpirams', verbose=False).run(
                _pekeris('mpirams', absorption), src, rcv,
                run_mode=RunMode.BROADBAND)
            H[label] = np.abs(np.asarray(field.data)).reshape(-1)
        assert H['plain'].size == freqs.size
        extra = 20.0 * np.log10(H['plain'] / H['lossy'])
        expected = self.BETA * 3000.0 / (1500.0 / freqs)
        assert np.all((0.8 < extra / expected) & (extra / expected < 1.35)), (
            f"broadband extra loss / plane-wave = {extra / expected}")



class TestRamsRangeStepStability:
    """The rams0.5 rotated Crank-Nicolson march diverges when its per-metre
    amplification of the steep propagating components just above cutoff
    outruns the seabed's leak of those components. The rule in
    ``pe_grid`` is pinned to the divergence table measured on a
    100 m sand channel (1700 m/s, 1.8 g/cm³, 0.5 dB/λ, c_s = 300 m/s): the
    binary was marched at each grid and the field either stayed finite or
    blew up within the first few hundred metres."""

    SAND = dict(c0=1591.0, water_sound_speed=1500.0, water_density=1.0,
                seabed_speed=1700.0, seabed_density=1.8,
                seabed_attenuation_dB_lambda=0.5, n_pade=6, theta_deg=45.0)
    #: (frequency Hz, dr m, water depth m, what the binary did)
    MEASURED = [(1000.0, 1.0, 100.0, 'ran'), (1500.0, 1.0, 100.0, 'diverged'),
                (1500.0, 0.5, 100.0, 'ran'), (1500.0, 0.25, 100.0, 'ran'),
                (2000.0, 0.5, 100.0, 'diverged'), (2000.0, 0.25, 100.0, 'ran'),
                (5000.0, 0.1, 100.0, 'diverged'), (5000.0, 0.04, 100.0, 'ran'),
                (1000.0, 1.0, 200.0, 'diverged'), (1000.0, 0.5, 200.0, 'ran'),
                (1000.0, 1.5, 50.0, 'ran')]

    def test_every_measured_divergence_has_positive_excess(self):
        from uacpy.models.pe_grid import rams_growth_margin
        for freq, dr, depth, outcome in self.MEASURED:
            excess = rams_growth_margin(dr, frequency=freq, depth=depth, safety=1.0,
                                        **self.SAND)['excess']
            if outcome == 'diverged':
                assert excess > 0.005, (freq, dr, depth, excess)
            else:
                # The fluid-Rayleigh leak is a lower bound on an elastic
                # seabed's, so a run that survived may sit a little above
                # zero; none sits far above.
                assert excess < 0.005, (freq, dr, depth, excess)

    def test_the_stable_step_sits_between_the_last_run_and_the_first_divergence(self):
        from uacpy.models.pe_grid import rams_stable_dr
        dr = rams_stable_dr(10.0, frequency=1500.0, depth=100.0, **self.SAND)
        assert 0.2 < dr < 0.5
        dr = rams_stable_dr(10.0, frequency=5000.0, depth=100.0, **self.SAND)
        assert 0.03 < dr < 0.1

    #: 100 m of 1500 m/s water over a 3000/1500 m/s limestone at 200 Hz, as
    #: RAM's band speeds put it: Lytaev c0 = 1897.4 m/s. Measured level drift
    #: of the rams march against Scooter (Kraken and OAST agree within
    #: 0.04 dB), depth and 1 km averaged over 1-10 km: -0.36 dB/km at
    #: dr = 1.5 m (the λ/5 cap), -0.093 dB/km at 0.75 m.
    LIMESTONE = dict(freq=200.0, c0=1897.4, c_min=1500.0, c_max_all=3000.0,
                     theta=45.0)

    def test_the_level_drift_predicts_the_measured_rams_loss(self):
        """The field's level is carried by its low-angle components, and
        the step's loss on the horizontal one reproduces the measured drift;
        the budget is scored on the largest rate over the band, which is at
        least that."""
        from uacpy.models.pe_grid import rotated_cn_growth
        model = RAM(verbose=False)
        lim = self.LIMESTONE
        k0 = 2 * np.pi * lim['freq'] / lim['c0']
        xi_horizontal = (lim['c0'] / lim['c_min']) ** 2 - 1.0
        for dr, measured in ((1.5, 0.36), (0.75, 0.093)):
            loss = -float(rotated_cn_growth(
                np.array([xi_horizontal]), k0, dr, 6, lim['theta'])[0]) \
                * 20.0 / np.log(10.0) * 1000.0
            assert loss == pytest.approx(measured, rel=0.05), (dr, loss)
            drift = ram_stability.rams_level_drift_dB(
                dr, max_range=1000.0, **lim, knobs=model._knob_record())
            assert drift >= loss

    def test_the_drift_budget_sits_between_the_returned_step_and_a_longer_one(
            self):
        """Both sides of :data:`RAMS_CN_DRIFT_BUDGET_DB`: the step returned
        for a 10 km march keeps the drift inside the budget, and a step 10 %
        longer exceeds it."""
        from uacpy.models.ram._domain import RAMS_CN_DRIFT_BUDGET_DB
        model = RAM(verbose=False)
        kw = dict(max_range=10000.0, **self.LIMESTONE)
        dr = ram_stability.rams_dr_for_level_drift(1.5, **kw,
                                                   knobs=model._knob_record())
        assert dr < 1.5
        assert ram_stability.rams_level_drift_dB(
            dr, **kw, knobs=model._knob_record()) <= RAMS_CN_DRIFT_BUDGET_DB
        assert (ram_stability.rams_level_drift_dB(dr / 0.9 * 1.001, **kw,
                                                  knobs=model._knob_record())
                > RAMS_CN_DRIFT_BUDGET_DB)
        # A march short enough to stay inside the budget keeps its step.
        assert ram_stability.rams_dr_for_level_drift(
            1.5, max_range=1000.0, **self.LIMESTONE,
            knobs=model._knob_record()) == 1.5

    def test_the_stability_helpers_name_rams_growth_margins_keywords(self):
        """``rams_stable_dr`` / ``rams_stable_theta`` spell out the
        keywords of ``rams_growth_margin``, so a misspelt one is refused at
        the call rather than deep inside the margin."""
        import inspect
        from uacpy.models.pe_grid import (
            rams_growth_margin, rams_stable_dr, rams_stable_theta)
        margin = [n for n in inspect.signature(rams_growth_margin).parameters
                  if n != 'dr']
        for fn in (rams_stable_dr, rams_stable_theta):
            params = inspect.signature(fn).parameters
            assert not any(p.kind is p.VAR_KEYWORD for p in params.values())
            assert [n for n in params if n != 'dr_max'] == margin
        with pytest.raises(TypeError, match='seabed_sped'):
            rams_stable_theta(frequency=1500.0, depth=100.0, seabed_sped=1.0,
                              **self.SAND)

    def test_growth_rises_as_the_cube_of_the_step_and_evanescent_components_decay(self):
        from uacpy.models.pe_grid import rotated_cn_growth
        xi = np.linspace(-0.999, -0.05, 2000)
        g1 = rotated_cn_growth(xi, 1.0, 1.0, 6, 45.0).max()
        g2 = rotated_cn_growth(xi, 1.0, 2.0, 6, 45.0).max()
        assert g2 / g1 == pytest.approx(4.0, rel=0.25)   # per metre: (k0 dr)³ / dr
        assert np.all(rotated_cn_growth(np.linspace(-300.0, -1.001, 500),
                                        4.0, 1.0, 6, 45.0) < 0.0)

    def test_the_rotation_floor_is_what_a_smaller_angle_or_higher_order_removes(self):
        from uacpy.models.pe_grid import rotated_growth_floor
        xi = np.linspace(-0.999, -0.05, 2000)
        k0 = 2 * np.pi * 1000.0 / 1591.0
        f45 = rotated_growth_floor(xi, k0, 6, 45.0).max()
        assert f45 == pytest.approx(3.7e-4, rel=0.1)
        assert rotated_growth_floor(xi, k0, 6, 20.0).max() < 1e-6
        assert rotated_growth_floor(xi, k0, 8, 45.0).max() < f45
        assert rotated_growth_floor(xi, k0, 6, 90.0).max() > 10 * f45

    def test_a_pinned_step_predicted_to_diverge_warns_before_the_march(self):
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              sound_speed=1700.0, density=1.8, attenuation=0.5,
                              shear_speed=300.0, shear_attenuation=1.0))
        ram = RAM(backend='rams', verbose=False)
        with pytest.warns(UserWarning, match='predicted to diverge') as rec:
            ram_collins.resolve_collins_grid(env, 1500.0, 'rams', 5000.0,
                                      1.0, 0.02, None, zs=30.0,
                                      knobs=ram._knob_record(), log=ram._log,
                                      speed_bounds=ram._speed_bounds)
        assert 'dr <= 0.3' in str(rec[0].message)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            ram_collins.resolve_collins_grid(env, 1500.0, 'rams', 5000.0,
                                      0.25, 0.02, None, zs=30.0,
                                      knobs=ram._knob_record(), log=ram._log,
                                      speed_bounds=ram._speed_bounds)

    def test_the_rule_does_not_apply_without_the_rotation(self):
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=_elastic_halfspace())
        model = RAM(backend='rams', rams_rotation=False,
                   verbose=False)
        assert ram_stability.rams_stability(
            env, 1500.0, dr=1.0, knobs=model._knob_record(),
            speed_bounds=model._speed_bounds) is None

    @pytest.mark.slow
    def test_the_divergence_warning_names_the_range_step(self):
        """Deck + binary at the first measured divergence (1.5 kHz, dr = 1 m):
        the samples come back NaN and the warning names the step that
        holds, not a Padé order or a depth step."""
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              sound_speed=1700.0, density=1.8, attenuation=0.5,
                              shear_speed=300.0, shear_attenuation=1.0))
        src = Source(depths=30.0, frequencies=1500.0)
        rcv = Receiver(depths=[30.0], ranges=[1000.0, 2000.0])
        with recorded_warnings() as rec:
            field = RAM(backend='rams', verbose=False, dr=1.0, dz=0.02).run(
                env, src, rcv, run_mode=RunMode.COHERENT_TL)
        assert np.isnan(np.asarray(field.tl, float)).all()
        texts = [str(w.message) for w in rec]
        assert any('predicted to diverge' in s for s in texts)
        assert any('Use dr <= 0.3' in s for s in texts)
        assert not any('larger n_pade or a finer dz' in s for s in texts)


class TestCollinsStopMessagesAreTypedErrors:
    """The Collins codes end their two self-diagnosed failures on a bare
    Fortran ``stop`` that exits 0: ``ramgeo1.5.f:138-149`` ("Need to
    increase parameter mz/mp/mr to N", before the march, empty ``tl.grid``)
    and ``ramgeo1.5.f:767-771`` ("Laguerre method not converging. Try a
    different combination of DR and NP.", mid-march, partial ``tl.grid``).
    ``ram.collins.raise_on_collins_stop`` turns either stdout line into a
    ``ModelExecutionError`` that quotes it and names the uacpy remedy. The
    subprocess is faked so the test does not depend on provoking a real
    stop."""

    @staticmethod
    def _run_with_stdout(monkeypatch, stdout):
        env = Environment(
            name='collins-stop', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=[50.0], ranges=[500.0, 1000.0])

        def fake_run(self, cmd, cwd=None, timeout=None, env=None):
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout,
                                               stderr='')
        monkeypatch.setattr(RAM, '_run_subprocess', fake_run)
        return RAM(backend='ramgeo', verbose=False, n_pade=6).run(
            env, src, rcv)

    def test_an_array_limit_stop_names_the_fortran_line_and_the_checker(
            self, monkeypatch):
        with pytest.raises(ModelExecutionError,
                           match='Need to increase parameter mp') as ei:
            self._run_with_stdout(
                monkeypatch, '   Need to increase parameter mp to 11\n')
        text = str(ei.value)
        assert 'Need to increase parameter mp to 11' in text
        assert 'bare Fortran STOP, exit code 0' in text
        assert ('uacpy.models.ram.collins.check_collins_array_limits'
                in text)
        assert 'ramgeo' in text

    def test_a_laguerre_stop_names_dr_and_n_pade(self, monkeypatch):
        with pytest.raises(ModelExecutionError,
                           match='Laguerre method not converging') as ei:
            self._run_with_stdout(
                monkeypatch,
                ' \n   Laguerre method not converging.\n'
                '   Try a different combination of DR and NP.\n \n')
        text = str(ei.value)
        assert 'Laguerre method not converging.' in text
        assert 'Try a different combination of DR and NP.' in text
        assert 'n_pade=6' in text and 'n_pade=4' in text
        # The remedy names the dr the run marched with, never the
        # constructor's unresolved None.
        assert 'dr=None' not in text
        assert re.search(r"RAM\(dr=\d+(\.\d+)?, n_pade=4\)", text), text

    def test_an_ordinary_silent_death_keeps_the_missing_output_error(
            self, monkeypatch):
        with pytest.raises(ModelExecutionError,
                           match='did not produce a TL grid') as ei:
            self._run_with_stdout(monkeypatch, '')
        text = str(ei.value)
        assert 'own diagnosis' not in text
        assert 'tl.grid' in text


class TestRamsStabilityRuleMeasuresXiAgainstK0:
    """``ξ = (k_r/k0)² − 1`` with ``k0 = ω/c0``, so a component's horizontal
    wavenumber against the WATER's is ``√(1+ξ)·c_w/c0`` and the seabed's
    critical angle sits at ``ξ = (c0/c_s)² − 1``. The rule had the ratio
    inverted, which is invisible at c0 = c_w and doubles the stable step's
    refinement over a 2400 m/s seabed (0.189 against 0.412 m)."""

    _SEABED = dict(water_sound_speed=1500.0, water_density=1.027,
                   seabed_speed=1700.0, seabed_density=1.8,
                   seabed_attenuation_dB_lambda=0.5, depth=100.0)

    def test_the_leak_is_the_rayleigh_loss_at_the_water_grazing_angle(self):
        from uacpy.models.pe_grid import seabed_leak_rate
        from uacpy.core.acoustics import reflection_coeff
        c0, xi = 1591.0, np.array([-0.6])
        grazing = np.arccos(np.sqrt(1.0 + xi) * 1500.0 / c0)
        r = np.abs(reflection_coeff(np.rad2deg(grazing), sound_speed=1700.0,
                                    density=1.8, attenuation=0.5,
                                    water_sound_speed=1500.0,
                                    water_density=1.027))
        want = -np.log(r) * np.tan(grazing) / 200.0
        got = seabed_leak_rate(xi, c0=c0, **self._SEABED)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_the_scored_band_stops_at_the_critical_angle_in_k0_units(self):
        from uacpy.models.pe_grid import _rams_steep_band
        c0 = 1591.0
        band = _rams_steep_band(c0, 1500.0, 1700.0)
        assert band.max() == pytest.approx((c0 / 1700.0) ** 2 - 1.0,
                                           abs=1e-3)

    def test_the_rule_flags_every_recorded_divergence(self):
        """The calibration's own grids (100 m and 200 m sand channels,
        1700 m/s, 1.8 g/cm³, 0.5 dB/λ, c_s = 300): growth/leak <= 0.98 where
        the march ran and >= 1.31 where it diverged, so ``safety = 2``
        predicts every divergence."""
        from uacpy import Environment, BoundaryProperties
        from uacpy.models.pe_grid import rams_growth_margin
        grids = [(100, 1000, 1.0, False), (100, 1500, 0.5, False),
                 (100, 2000, 0.25, False), (100, 5000, 0.04, False),
                 (100, 1500, 1.0, True), (100, 2000, 0.5, True),
                 (100, 5000, 0.1, True), (200, 1000, 1.0, True)]
        ram = RAM()
        for depth, f, dr, diverged in grids:
            env = Environment(bathymetry=float(depth), ssp=1500.0,
                              bottom=BoundaryProperties(
                                  acoustic_type='half-space',
                                  sound_speed=1700.0, density=1.8,
                                  attenuation=0.5, shear_speed=300.0))
            params = ram_stability.rams_stability_params(
                env, float(f), knobs=ram._knob_record(),
                speed_bounds=ram._speed_bounds)[0]
            m = rams_growth_margin(dr, safety=1.0, **params)
            ratio = m['growth'] / m['leak']
            if diverged:
                assert ratio > 1.2, (depth, f, dr, ratio)
                assert rams_growth_margin(dr, **params)['excess'] > 0.0
            else:
                assert ratio < 1.0, (depth, f, dr, ratio)


class TestRamFollowsTheLinearSspBetweenDeclaredProfiles:
    """Every RAM backend marches the nearest written profile, while the
    carrier (``SoundSpeedProfile.eval``), Bellhop and Kraken interpolate
    linearly. Writing only the declared profiles turned a 10 km front into a
    step at 5 km: 3.3 dB median off the same front written densely."""

    @staticmethod
    def _front(n_columns):
        from uacpy import Environment, BoundaryProperties
        from uacpy import SoundSpeedProfile
        z = np.array([0.0, 30.0, 100.0])
        c0 = np.array([1500.0, 1500.0, 1500.0])
        c1 = np.array([1530.0, 1490.0, 1485.0])
        r = np.linspace(0.0, 10000.0, n_columns)
        m = np.column_stack([c0 + (c1 - c0) * x / 10000.0 for x in r])
        return Environment(
            bathymetry=100.0, ssp=SoundSpeedProfile.from_2d(z, r, m),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5))

    def test_the_written_axis_steps_at_most_the_allowed_speed(self):
        from uacpy.models.ram._domain import _SSP_STEP_MAX_MPS
        env = self._front(2)
        axis = ram_domain.ssp_range_axis(env)
        assert axis[0] == 0.0 and axis[-1] == 10000.0
        cols = np.column_stack([
            np.interp([0.0, 30.0, 100.0], *env.ssp.eval(range=float(r))
                      .to_pairs().T) for r in axis])
        assert np.max(np.abs(np.diff(cols, axis=1))) <= _SSP_STEP_MAX_MPS + 1e-9

    def test_the_profile_count_is_capped(self):
        from uacpy import Environment, SoundSpeedProfile
        from uacpy.models.ram._domain import _MAX_SSP_PROFILES
        z = np.array([0.0, 100.0])
        m = np.array([[1450.0, 1550.0], [1450.0, 1550.0]])   # 100 m/s step
        env = Environment(bathymetry=100.0, ssp=SoundSpeedProfile.from_2d(
            z, np.array([0.0, 1000.0]), m))
        assert ram_domain.ssp_range_axis(env).size <= _MAX_SSP_PROFILES

    @pytest.mark.parametrize('backend', ['ramgeo', 'mpirams'])
    def test_two_declared_profiles_march_like_the_dense_front(self, backend):
        from uacpy import Source, Receiver
        src = Source(depths=20.0, frequencies=200.0)
        rcv = Receiver(depths=np.linspace(5.0, 95.0, 19),
                       ranges=np.linspace(1000.0, 10000.0, 19))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sparse = RAM(backend=backend).compute_tl(self._front(2), src, rcv)
            dense = RAM(backend=backend).compute_tl(self._front(41), src, rcv)
        d = np.abs(np.asarray(sparse.dB) - np.asarray(dense.dB))
        assert float(np.nanmedian(d)) < 0.2


def test_ram_models_the_same_flat_earth_as_every_other_engine_by_default():
    """EXPERT-6 / Theo's decision: RAM alone applied Earth flattening by
    default, which moved its convergence zones against every other engine's.
    It is off by default and stays available."""
    assert RAM().earth_curvature is False
    assert RAM(earth_curvature=True).earth_curvature is True


def test_a_frequencies_override_keeps_the_sources_level():
    """``run(frequencies=…)`` replaces only the Source's frequency axis:
    ``source_level_dB`` reaches the result exactly as it does when
    the same band is given on the Source itself."""
    env = Environment(
        name='level', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1600.0, density=1.5,
                                  attenuation=0.5))
    rcv = Receiver(depths=[30.0], ranges=[1000.0, 2000.0])
    band = np.linspace(98.0, 102.0, 5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        override = RAM(verbose=False).run(
            env, Source(depths=30.0, frequencies=100.0, source_level_dB=180.0),
            rcv, run_mode=RunMode.BROADBAND, frequencies=band)
        on_source = RAM(verbose=False).run(
            env, Source(depths=30.0, frequencies=band, source_level_dB=180.0),
            rcv, run_mode=RunMode.BROADBAND)
    assert on_source.source_level_dB == 180.0
    assert override.source_level_dB == 180.0
    assert np.allclose(override.frequencies, band)


def _pekeris_100m():
    return make_pekeris(name='pekeris', sound_speed=1600.0, density=1.5)


def test_mpirams_coherent_tl_marches_one_bin_whatever_q_and_t_hold(
        monkeypatch):
    """The TL field keeps only the centre bin, so a Q/T pinned for
    broadband runs must not widen the COHERENT_TL sweep: the deck is
    written with (Q, T) = (1e6, 1) and the field matches an unpinned run."""
    seen = []
    original = ram_mpirams.write_mpirams_deck

    def spy(inputs, **kwargs):
        engine = inputs.settings.engine
        seen.append((engine.q_factor, engine.record_duration))
        return original(inputs, **kwargs)

    monkeypatch.setattr(
        'uacpy.models.ram._model.write_mpirams_deck', spy)
    src = Source(depths=30.0, frequencies=100.0)
    rcv = Receiver(depths=[30.0, 60.0], ranges=[1000.0, 2000.0, 3000.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pinned = RAM(q_factor=2.0, record_duration=2.0, verbose=False).compute_tl(
            _pekeris_100m(), src, rcv)
        free = RAM(verbose=False).compute_tl(_pekeris_100m(), src, rcv)
    assert seen == [(1e6, 1.0), (1e6, 1.0)]
    assert np.array_equal(np.asarray(pinned.data), np.asarray(free.data))


def test_a_collins_broadband_marches_a_non_uniform_band_bin_for_bin():
    """ramgeo's BROADBAND is one subprocess per bin, so an increasing
    non-uniform band (octave centres) runs and comes back on exactly those
    bins; mpiramS, whose sweep is uniform, refuses the same band."""
    band = np.array([50.0, 100.0, 200.0])
    src = Source(depths=30.0, frequencies=band)
    rcv = Receiver(depths=[30.0], ranges=[1000.0, 2000.0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        field = RAM(backend='ramgeo', verbose=False).run(
            _pekeris_100m(), src, rcv, run_mode=RunMode.BROADBAND)
    assert np.array_equal(field.frequencies, band)
    assert np.all(np.isfinite(np.asarray(field.data)))
    assert field.run_settings.engine.q_factor is None and field.run_settings.engine.df_hz is None
    with pytest.raises(ConfigurationError, match='non-uniform'):
        RAM(backend='mpirams', verbose=False).run(
            _pekeris_100m(), src, rcv, run_mode=RunMode.BROADBAND)


def test_a_single_frequency_source_broadband_is_the_shared_default_band():
    """A lone Source frequency on BROADBAND expands to the base default
    band — ``DEFAULT_BROADBAND_N_FREQS`` bins over
    ``fc·(1 ± DEFAULT_BROADBAND_BANDWIDTH_FACTOR/2)`` — the axis Bellhop,
    Kraken and Scooter return for the same call; ``run(frequencies=[fc])``
    stays the one-bin request."""
    from uacpy.models._defaults import (
        DEFAULT_BROADBAND_BANDWIDTH_FACTOR, DEFAULT_BROADBAND_N_FREQS,
    )
    src = Source(depths=30.0, frequencies=100.0)
    rcv = Receiver(depths=[30.0], ranges=[1000.0, 2000.0])
    model = RAM(backend='mpirams', verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        band = model.run(_pekeris_100m(), src, rcv,
                         run_mode=RunMode.BROADBAND)
        one = model.run(_pekeris_100m(), src, rcv,
                        run_mode=RunMode.BROADBAND, frequencies=[100.0])
    half = 0.5 * DEFAULT_BROADBAND_BANDWIDTH_FACTOR
    np.testing.assert_allclose(
        band.frequencies,
        np.linspace(100.0 * (1 - half), 100.0 * (1 + half),
                    DEFAULT_BROADBAND_N_FREQS))
    assert np.all(np.isfinite(np.asarray(band.data)))
    np.testing.assert_array_equal(one.frequencies, [100.0])


def _francois_garrison_env():
    return Environment(
        name='fg', bathymetry=100.0, ssp=[(0.0, 1500.0), (100.0, 1480.0)],
        absorption=FrancoisGarrison(temperature=10.0, salinity=35.0,
                                    pH=8.0),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1600.0, density=1.5,
                                  attenuation=0.5))


@pytest.mark.parametrize('backend', ['mpirams', 'ramgeo'])
def test_a_sub_200_hz_band_gives_one_absorption_range_notice(backend):
    """The water alpha of a broadband sweep is evaluated in one
    ``absorption.alpha`` call, so Francois-Garrison's out-of-range notice
    (fitted from 200 Hz) fires once for a 21-bin 50-150 Hz band, not once
    per bin."""
    src = Source(depths=30.0, frequencies=np.linspace(50.0, 150.0, 21))
    rcv = Receiver(depths=[30.0], ranges=[1000.0])
    with recorded_warnings() as caught:
        RAM(backend=backend, verbose=False).run(
            _francois_garrison_env(), src, rcv, run_mode=RunMode.BROADBAND)
    notices = [w for w in caught if 'FrancoisGarrison' in str(w.message)]
    assert len(notices) == 1, [str(w.message) for w in notices]
    assert '21 of 21' in str(notices[0].message)


@pytest.mark.parametrize('dz', [None, 0.5, 7.0])
def test_the_band_table_gives_the_per_bin_water_block_bit_for_bit(dz):
    env = _francois_garrison_env()
    freqs = np.linspace(50.0, 150.0, 21)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        band = ram_domain.water_alpha_band(env, freqs, 150.0)
        for f in freqs:
            assert (ram_domain.water_attenuation_block(
                        env, f, env.ssp.to_pairs(), 150.0, dz, band=band)
                    == ram_domain.water_attenuation_block(
                        env, f, env.ssp.to_pairs(), 150.0, dz))


class TestCollinsDecksCarryOnlyTheMarchedTrack:
    """The Collins binaries move to the next bathymetry node only once the
    march reaches it (``ramgeo1.5.f:348``), so nodes past the first one
    beyond the farthest receiver never shape the returned field; the deck
    omits them so the 505-node array guard and the ``dr`` bound see only
    what the binary consumes."""

    def test_nodes_to_the_last_range_and_one_beyond_are_kept(self):
        nodes = [(0.0, 100.0), (1000.0, 90.0), (3000.0, 80.0),
                 (3500.0, 70.0), (4000.0, 60.0)]
        assert ram_collins.within_march(nodes, 3000.0) == nodes[:4]
        assert ram_collins.within_march(nodes, 2999.0) == nodes[:3]

    @pytest.mark.requires_binary
    def test_a_long_track_runs_a_short_march(self):
        """Measured before: 600 nodes over 60 km with receivers to 3 km were
        refused ('the binary's arrays hold 505')."""
        r = np.linspace(0.0, 60000.0, 600)
        env = uacpy.Environment(
            bathymetry=np.column_stack([r, 100.0 + 0.0005 * r]), ssp=1500.0)
        rcv = uacpy.Receiver(depths=[30.0], ranges=np.linspace(200.0, 3000.0, 8))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = RAM(backend='ramgeo').compute_tl(
                env, uacpy.Source(depths=50.0, frequencies=100.0), rcv)
        assert np.isfinite(np.asarray(result.tl)).all()


# ─── The run goes through the stages: settings resolved once, decks from them ─


def _pekeris_guide(depth=100.0, **kw):
    return Environment(
        bathymetry=depth, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1700.0, density=1.5,
                                  attenuation=0.5), **kw)


def _notices(call):
    """The warning texts ``call()`` emits, and what it returned."""
    with recorded_warnings() as rec:
        out = call()
    return [str(w.message) for w in rec], out


class TestTheDecksAreWrittenFromTheResolvedSettings:
    """``run_settings().engine`` is a ``RamSettings``: the backend and the PE
    grid of every launch, resolved once. The decks carry exactly those
    values, and the result reports the same grid."""

    SRC = Source(depths=50.0, frequencies=100.0)
    RCV = Receiver(depths=[20.0, 60.0], ranges=[500.0, 1500.0])

    def test_mpirams_in_pe_carries_the_settings_grid(self, tmp_path):
        model = RAM(verbose=False, work_dir=str(tmp_path), cleanup=False)
        settings = model.run_settings(_pekeris_guide(), self.SRC, self.RCV)
        grid = settings.engine.grids[0]
        field = model.run(_pekeris_guide(), self.SRC, self.RCV)
        lines = (tmp_path / 'in.pe').read_text().split('\n')
        assert (float(lines[4]), float(lines[5])) == (grid.dz, grid.dr)
        assert (field.run_settings.engine.grids[0].dr, field.run_settings.engine.grids[0].dz,
                field.run_settings.engine.grids[-1].zmax) == (grid.dr, grid.dz, grid.zmax)
        assert field.run_settings == settings

    def test_a_collins_deck_carries_the_settings_grid(self, tmp_path):
        model = RAM(backend='ramgeo', verbose=False, work_dir=str(tmp_path),
                    cleanup=False)
        settings = model.run_settings(_pekeris_guide(), self.SRC, self.RCV)
        grid = settings.engine.grids[0]
        field = model.run(_pekeris_guide(), self.SRC, self.RCV)
        row = (tmp_path / 'ramgeo.in').read_text().split('\n')[2].split()
        assert [float(v) for v in row] == pytest.approx(
            [grid.rmax_march, grid.dr, grid.ndr], rel=1e-11)
        assert (field.run_settings.engine.grids[0].dr, field.run_settings.engine.grids[0].dz,
                field.run_settings.engine.grids[-1].zmax) == (grid.dr, grid.dz, grid.zmax)
        assert field.run_settings == settings

    def test_a_collins_band_resolves_one_grid_per_bin(self):
        settings = RAM(backend='ramgeo', verbose=False).run_settings(
            _pekeris_guide(), self.SRC, self.RCV, RunMode.BROADBAND,
            frequencies=[90.0, 100.0, 110.0])
        engine = settings.engine
        assert [g.frequency for g in engine.grids] == [90.0, 100.0, 110.0]
        assert len({(g.dr, g.dz, g.zmax) for g in engine.grids}) == 1
        assert engine.marched_frequencies.tolist() == [90.0, 100.0, 110.0]

    def test_the_settings_round_trip_and_print_the_grid(self):
        import pickle
        from uacpy.core.run_settings import RunSettings
        settings = RAM(verbose=False).run_settings(
            _pekeris_guide(), self.SRC, self.RCV)
        assert RunSettings.from_dict(settings.to_dict()) == settings
        assert pickle.loads(pickle.dumps(settings)) == settings
        assert 'depth points' in repr(settings)


class TestStageThreeNoticesComeFromRunAndRunSettingsOnly:
    """A grid notice is recorded in the settings and emitted by ``run`` and
    ``run_settings``; ``validate_inputs`` states only what is refused."""

    def test_validate_inputs_is_silent_where_run_settings_warns(self):
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=[20.0], ranges=[1000.0])
        model = RAM(verbose=False, dz=1.0)
        said, settings = _notices(lambda: model.run_settings(
            _pekeris_guide(), src, rcv))
        assert any('of a cell from the quarter cell' in t for t in said)
        assert any('of a cell from the quarter cell' in t for t in (x.message for x in settings.engine.notices))
        said, _ = _notices(lambda: model.validate_inputs(_pekeris_guide(), src, rcv))
        assert not any('of a cell from the quarter cell' in t for t in said)

    def test_validate_inputs_refuses_a_time_series_without_a_pulse_like_run(
            self, monkeypatch):
        model = RAM(verbose=False)
        monkeypatch.setattr(model, '_run_subprocess',
                            lambda *a, **k: pytest.fail('launched'))
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=[20.0], ranges=[1000.0])
        messages = set()
        for call in (model.validate_inputs, model.run_settings, model.run):
            with pytest.raises(ConfigurationError,
                               match='requires source_waveform') as info:
                call(_pekeris_guide(), src, rcv, RunMode.TIME_SERIES)
            messages.add(str(info.value))
        assert len(messages) == 1

    def test_a_two_depth_call_warns_once(self):
        """Stages 1-3 run once per call: a keyword the mode drops is said
        once, not once per source depth."""
        said, _ = _notices(lambda: RAM(verbose=False).run(
            _pekeris_guide(), Source(depths=[30.0, 60.0], frequencies=100.0),
            Receiver(depths=[20.0], ranges=[1000.0]),
            source_waveform=np.hanning(40), sample_rate=400.0))
        assert len([t for t in said if 'ignoring source_waveform' in t]) == 1


class TestAPinnedDzFarFromTheQuarterCellWarns:
    """B6 / RAM-2: on the fluid backends a pinned ``dz`` that puts the
    shallowest seafloor more than ``SEAFLOOR_QUARTER_TOLERANCE`` (0.1) of a
    cell from the quarter-cell placement warns and names the aligned value
    ``h/(n + 1/4)``; the ``dz`` is marched as given. The rule reads the
    distance round the cell, so a node and a placement past mid-cell both
    warn (mpiramS against Kraken, 120 m / 120 Hz: 0.49 dB at the quarter,
    0.78-0.97 at 0.075 cell, 1.26-1.40 at 0.125, 2.84 on the node, 4.56 at
    0.95 of a cell; ``log/fix-r2-models_scratch/r2_ram_placement``). rams0.5
    wants the node and is not checked."""

    SRC = Source(depths=50.0, frequencies=100.0)
    RCV = Receiver(depths=[20.0], ranges=[1000.0])

    def _said(self, **kw):
        said, settings = _notices(lambda: RAM(verbose=False, **kw)
                                  .run_settings(_pekeris_guide(), self.SRC,
                                                self.RCV))
        return [t for t in said if 'of a cell from the quarter cell' in t], \
            settings

    @pytest.mark.parametrize('dz, aligned', [(1.0, '0.997506'),
                                             (0.5, '0.499376'),
                                             (0.25, '0.249844')])
    def test_a_round_dz_warns_naming_the_aligned_value(self, dz, aligned):
        said, settings = self._said(dz=dz)
        assert len(said) == 1 and f"dz={aligned} m" in said[0]
        assert settings.engine.grids[0].dz == dz

    def test_the_aligned_value_is_silent(self):
        assert self._said(dz=100.0 / 100.25)[0] == []

    @pytest.mark.parametrize('frac', [0.6, 0.75, 0.95])
    def test_a_seafloor_past_mid_cell_warns(self, frac):
        """Off the node and still far from the quarter: the placements the
        on-node rule left silent, measured 3.1-4.6 dB rms."""
        said, _ = self._said(dz=100.0 / (100.0 + frac))
        assert len(said) == 1 and f"{frac:.3g} of a cell below" in said[0]

    @pytest.mark.parametrize('frac, warns', [(0.1499, True),
                                             (0.1501, False),
                                             (0.3499, False),
                                             (0.3501, True)])
    def test_the_tolerance_is_a_tenth_of_a_cell_either_side(self, frac,
                                                            warns):
        """frac 0.15 and 0.35 are 0.1 cell above and below the quarter."""
        from uacpy.models.ram.grid import SEAFLOOR_QUARTER_TOLERANCE
        assert SEAFLOOR_QUARTER_TOLERANCE == 0.1
        said, _ = self._said(dz=100.0 / (100.0 + frac))
        assert bool(said) is warns

    @pytest.mark.parametrize('backend', ['ramgeo', 'mpirams'])
    def test_every_fluid_backend_warns(self, backend):
        assert self._said(dz=1.0, backend=backend)[0]


class TestRamsurfPutsTheSurfaceOnADepthNode:
    """RAM-1: ramsurf1.5 zeroes rows down to ``int(1 + zsrf/dz)``, so its
    surface sits at ``floor(zsrf/dz)·dz``. The automatic grid puts the
    surface at r = 0 on a node as well as the seafloor a quarter cell below
    one; a pinned dz that misses the node, and a relief of few cells, warn."""

    SRC = Source(depths=50.0, frequencies=100.0)
    RCV = Receiver(depths=[20.0, 60.0], ranges=[1000.0, 3000.0])

    @staticmethod
    def _env(depression):
        return _pekeris_guide(altimetry=[(0.0, -depression), (6000.0, -depression)])

    def _settings(self, depression, **kw):
        return _notices(lambda: RAM(verbose=False, **kw).run_settings(
            self._env(depression), self.SRC, self.RCV))

    @pytest.mark.parametrize('depression', [10.0, 7.3, 5.0])
    def test_the_automatic_dz_puts_the_surface_on_a_node(self, depression):
        said, settings = self._settings(depression)
        dz = settings.engine.grids[0].dz
        assert settings.engine.backend == 'ramsurf'
        frac = depression / dz - np.floor(depression / dz)
        assert 1e-6 <= frac <= 0.05
        # ... with the seafloor where the fluid codes want it.
        assert 100.0 / dz - np.floor(100.0 / dz) == pytest.approx(0.25,
                                                                  abs=1e-6)
        assert not [t for t in said if 'sits at' in t]

    def test_a_pinned_dz_off_the_node_warns_where_the_surface_sits(self):
        said, _ = self._settings(10.0, dz=0.9412)
        hits = [t for t in said if 'sits at 9.412 m' in t]
        assert len(hits) == 1 and 'asked at 10 m' in hits[0]

    def test_a_pinned_dz_on_the_node_is_silent(self):
        said, _ = self._settings(10.0, dz=1.0 / 1.0000001 * 100.0 / 100.25)
        assert not [t for t in said if 'sits at' in t]

    @pytest.mark.parametrize('cells, warns', [(3.99, True), (4.01, False)])
    def test_a_relief_of_few_cells_warns(self, cells, warns):
        said, _ = self._settings(4.0, dz=4.0 / cells)
        assert bool([t for t in said if 'depth cells' in t]) is warns

    def test_the_misfit_counts_a_ratio_just_under_an_integer_as_a_cell(self):
        misfit = ram_grid.surface_node_misfit
        assert misfit(10.0, 1.0) == 0.0
        assert misfit(10.0, 10.0 / 9.9999999) == 1.0
        assert misfit(10.0, 0.9) == pytest.approx(10.0 / 0.9 - 11.0)


class TestAFluidColumnBesideAnElasticOneIsRefused:
    """RAM-5: rams0.5 divides by the shear modulus of the first node below
    the seafloor, so a range-dependent seabed mixing an elastic column with a
    layer-free fluid half-space came back NaN past the switch. It is refused
    by all three entry points, before any launch."""

    @staticmethod
    def _env(fluid_shear):
        bottom = Bottom(columns=[
            SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0, density=2.0,
                attenuation=0.5, shear_speed=400.0, shear_attenuation=1.0)),
            SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0, density=1.8,
                attenuation=0.5, shear_speed=fluid_shear))],
            ranges=[0.0, 3000.0])
        return Environment(bathymetry=100.0, ssp=1500.0, bottom=bottom)

    def test_every_entry_point_refuses_it_naming_the_fluid_column(
            self, monkeypatch):
        model = RAM(verbose=False)
        monkeypatch.setattr(model, '_run_subprocess',
                            lambda *a, **k: pytest.fail('launched'))
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=[50.0], ranges=[1000.0, 4000.0])
        messages = set()
        for call in (model.validate_inputs, model.run_settings, model.run):
            with pytest.raises(ConfigurationError,
                               match='column 1 has no layers') as info:
                call(self._env(0.0), src, rcv)
            messages.add(str(info.value))
        assert len(messages) == 1

    def test_an_elastic_column_on_both_sides_runs_on_rams(self):
        settings = RAM(verbose=False).run_settings(
            self._env(300.0), Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[50.0], ranges=[1000.0]))
        assert settings.engine.backend == 'rams'


class TestALonePinnedBandKnobKeepsTheDefaultBand:
    """RAM-4: with one of ``Q`` / ``T`` pinned on a lone source frequency,
    the other takes the value of the default band every engine shares
    (128 bins over fc·(1 ± 1/4)), not a RAM-private one."""

    def test_a_pinned_t_takes_the_default_half_width(self):
        model = RAM(verbose=False, record_duration=1.0)
        fc, Q, T = ram_band.resolve_broadband_grid(
            Source(depths=50.0, frequencies=1000.0),
            knobs=model._knob_record(), log=model._log)
        assert (fc, Q, T) == (1000.0, 4.0, 1.0)
        frq = ram_band.broadband_frequencies(fc, Q, T)
        assert (frq[0], frq[-1], frq.size) == (750.0, 1250.0, 501)

    def test_a_pinned_q_takes_the_default_spacing(self):
        model = RAM(verbose=False, q_factor=8.0)
        fc, Q, T = ram_band.resolve_broadband_grid(
            Source(depths=50.0, frequencies=1000.0),
            knobs=model._knob_record(), log=model._log)
        assert Q == 8.0 and 1.0 / T == pytest.approx(1000.0 * 0.5 / 127.0)


class TestADescendingBandIsRefusedAsSuch:
    """RAM-15: a frequency array that does not increase is refused saying
    so, not as a 'degenerate range'."""

    def test_it_names_the_first_step_out_of_order(self):
        with pytest.raises(ConfigurationError,
                           match='strictly increasing; frequencies'
                                 r'\[0\]=300 Hz is followed by 200 Hz'):
            model = RAM(verbose=False)
            ram_band.resolve_broadband_grid(
                Source(depths=50.0, frequencies=[300.0, 200.0, 100.0]),
                knobs=model._knob_record(), log=model._log)


class TestTheBottomSwitchesAtItsOwnMidpoint:
    """RAM-14: with SSP breaks between two bottom breaks, the Collins
    sections and the mpiramS sediment profiles put the column switch where
    ``Bottom.at`` does — midway between the bottom's own breaks."""

    @staticmethod
    def _env():
        bottom = Bottom(columns=[
            SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0, density=1.6,
                attenuation=0.5)),
            SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0, density=2.0,
                attenuation=0.5))], ranges=[0.0, 4000.0])
        breaks = [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
        c = [1500.0 + 2.0 * r / 5000.0 for r in breaks]
        return Environment(bathymetry=100.0, bottom=bottom,
                           ssp=SoundSpeedProfile(depths=[0.0, 100.0],
                                                 sound_speed=[c, c],
                                                 ranges=breaks))

    def test_a_collins_deck_switches_at_2000_m(self):
        base = ram_collins.collins_deck_base(
            self._env(), 'ramgeo', 400.0,
            knobs=RAM(verbose=False)._knob_record())
        speeds = [(seg['range'], seg['bottom_c'][0][1])
                  for seg in base['segments']]
        before = [c for r, c in speeds if r < 2000.0]
        after = [c for r, c in speeds if r >= 2000.0]
        assert 2000.0 in [r for r, _ in speeds]
        assert set(before) == {1600.0} and set(after) == {1800.0}

    def test_the_mpirams_profiles_bracket_the_switch(self):
        ranges = ram_mpirams.sediment_profile_ranges(
            self._env(), log=RAM(verbose=False)._log)
        assert 1999.5 in ranges and 2000.5 in ranges and 2000.0 not in ranges

    def test_a_switch_already_on_a_section_edge_adds_nothing(self):
        env = self._env()
        env.ssp = SoundSpeedProfile(depths=[0.0, 100.0],
                                    sound_speed=[1500.0, 1500.0])
        base = ram_collins.collins_deck_base(
            env, 'ramgeo', 400.0, knobs=RAM(verbose=False)._knob_record())
        assert [seg['range'] for seg in base['segments']] == [0.0, 2000.0]


class TestAWideBandSaysWhatItsAbsorberCosts:
    """RAM-6: a band marches one grid, dz at its top and the absorber at its
    bottom; past ABSORBER_COST_NOTICE_POINTS points held mostly by the
    absorber the run warns, below it logs."""

    @pytest.mark.parametrize('points, warns', [(10000, False),
                                               (10001, True)])
    def test_it_warns_past_the_depth_budget(self, points, warns):
        from uacpy.models.ram import RamGrid
        assert ram_band.ABSORBER_COST_NOTICE_POINTS == 10000
        model = RAM(verbose=False)
        env = _pekeris_guide()
        zmax = 4.0 * ram_domain.absorbing_layer_thickness(
            env, 20.0, knobs=model._knob_record(),
            speed_bounds=model._speed_bounds) / 3.0
        grid = RamGrid(frequency=100.0, dr=10.0, dz=zmax / points,
                       zmax=zmax, n_depth_points=points)
        said, _ = _notices(lambda: ram_band.report_band_grid_cost(
            env, 'mpirams', grid, 20.0, 500.0, knobs=model._knob_record(),
            log=model._log, speed_bounds=model._speed_bounds))
        assert bool([t for t in said if 'absorbing layer' in t]) is warns

    def test_a_default_time_series_of_a_ricker_warns(self):
        fs = 2000.0
        t = np.arange(-0.02, 0.02, 1.0 / fs)
        a = (np.pi * 200.0 * t) ** 2
        said, settings = _notices(lambda: RAM(verbose=False).run_settings(
            _pekeris_guide(), Source(depths=50.0, frequencies=200.0),
            Receiver(depths=[50.0], ranges=[1000.0]), RunMode.TIME_SERIES,
            source_waveform=(1 - 2 * a) * np.exp(-a), sample_rate=fs,
            output_duration=0.5))
        assert settings.engine.grids[0].n_depth_points > 10000
        assert [t for t in said if 'absorbing layer' in t]


class TestOneWarningPerRunForABand:
    """RAM-17: the Collins band says once what it would otherwise say once
    per bin."""

    def test_diverged_bins_are_reported_together(self):
        reports = [dict(n_invalid=2, size=10, frequency=f, detail='d')
                   for f in (100.0, 110.0)]
        said, _ = _notices(
            lambda: ram_collins.warn_on_diverged_collins_samples(
                'ramgeo', reports, 5))
        assert len(said) == 1
        assert 'at 2 of 5 frequencies (100.00-110.00 Hz)' in said[0]
        assert '4/20 TL samples' in said[0]

    def test_one_diverged_bin_keeps_its_own_wording(self):
        said, _ = _notices(
            lambda: ram_collins.warn_on_diverged_collins_samples(
                'ramgeo', [dict(n_invalid=2, size=10, frequency=100.0,
                                detail='d')], 5))
        assert said == [
            "RAM:ramgeo: 2/10 TL samples at f=100.00 Hz are NaN/inf or "
            "below the level their range allows (Padé instability or PE "
            "divergence) and are returned as NaN — no data there, not a "
            "shadow zone. Every other sample is the march's own value. d"]


@pytest.mark.parametrize('backend', ['mpirams', 'ramgeo'])
@pytest.mark.parametrize('mode', [RunMode.COHERENT_TL, RunMode.BROADBAND])
def test_no_setting_is_copied_into_the_metadata(backend, mode):
    """The grid, the Padé point and the band are the run settings'
    (decision 4), on both families and both result paths."""
    kw = {} if mode is RunMode.COHERENT_TL else {'frequencies': [90.0, 100.0,
                                                                110.0]}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        field = RAM(verbose=False, backend=backend).run(
            _pekeris_guide(), Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[50.0], ranges=[1000.0]), mode, **kw)
    copies = {'dr', 'dz', 'zmax', 'pe_reference_speed', 'c0', 'q_factor', 'record_duration',
              'bandwidth_hz', 'df_hz'}
    assert not copies & set(field.metadata), sorted(field.metadata)
    assert field.run_settings.engine.grids[0].dz > 0.0


class TestTheBroadbandMetadataIsOneVocabulary:
    """RAM-13: mpiramS and the Collins backends record the band in the same
    settings fields (Q, T, bandwidth_hz, df_hz), not as metadata; the stock
    driver's 4·fc bookkeeping rate is not stamped as a sample rate."""

    @pytest.mark.parametrize('backend', ['mpirams', 'ramgeo'])
    def test_both_families_stamp_the_same_band_keys(self, backend):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = RAM(verbose=False, backend=backend).run(
                _pekeris_guide(), Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[50.0], ranges=[1000.0]), RunMode.BROADBAND,
                frequencies=[90.0, 100.0, 110.0])
        md, engine = field.metadata, field.run_settings.engine
        assert (engine.record_duration, engine.df_hz, engine.bandwidth_hz) == pytest.approx(
            (0.1, 10.0, 30.0))
        assert not {'fs', 'record_duration', 'df_hz', 'bandwidth_hz'} & set(md)


def test_ram_reads_c_max_and_rot0_from_one_decider_each():
    """RAM-11 / RAM-10: ``c_max`` is the base's (no fallback to the PE
    reference speed), and rams' carrier ``rot0`` is ``pe_grid``'s
    transcription of ``rpade``, the one the stability rule reads."""
    from uacpy.models.base import PropagationModel
    from uacpy.models.pe_grid import rotated_pade_coefficients
    assert RAM._speed_bounds is PropagationModel._speed_bounds
    model = RAM(verbose=False, n_pade=4)
    assert ram_stability.rams_rot0(
        30.0, knobs=model._knob_record()) == rotated_pade_coefficients(4,
                                                                       30.0)[2]


@pytest.mark.requires_binary
class TestRAM:
    """Tests for RAM model (mpiramS backend)."""

    def test_ram_returns_finite_tl_grid(self, simple_env, source, receiver_small):
        """Test RAM TL computation."""
        ram = RAM(verbose=False, dr=20.0, dz=2.0)
        result = ram.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert isinstance(result, Field)
        assert result.shape[0] > 0  # Has depth dimension
        assert result.shape[1] > 0  # Has range dimension
        assert np.all(np.isfinite(result.data))

    def test_ram_broadband_mode(self, simple_env, source):
        """RAM BROADBAND returns the H(f) transfer function."""
        ram = RAM(q_factor=2.0, record_duration=2.0, dr=20.0, dz=2.0, verbose=False)
        receiver = Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([5000.0])
        )
        result = ram.run(
            simple_env, source, receiver,
            run_mode=RunMode.BROADBAND
        )
        assert isinstance(result, Field)
        assert np.iscomplexobj(result.data)
        # Shape: (n_d, n_r, n_f) — trailing axis is the
        # variable dimension (frequency, here).
        assert result.data.shape[0] > 0  # depth
        assert result.data.shape[1] > 0  # range
        # mpiramS builds its grid as frq = fc + [-nf1..nf1]·df with
        # df = 1/T and nf1 = int((fc/Q - df)/df) + 1 (peramx.f90:353-383):
        # fc=100, q_factor=2, record_duration=2 → df=0.5, nf1=100, nf=201 spanning 50-150 Hz.
        f = np.asarray(result.coords['frequency'], dtype=float)
        assert result.data.shape[2] == 201
        assert f[0] == pytest.approx(50.0)
        assert f[-1] == pytest.approx(150.0)
        assert np.allclose(np.diff(f), 0.5, atol=1e-6)
        assert f[100] == pytest.approx(100.0)

    def test_ram_time_series_requires_waveform(self, simple_env, source):
        """TIME_SERIES without source_waveform must raise."""
        ram = RAM(q_factor=2.0, record_duration=2.0, dr=20.0, dz=2.0, verbose=False)
        receiver = Receiver(
            depths=np.array([50.0]),
            ranges=np.array([5000.0])
        )
        with pytest.raises(ConfigurationError, match="source_waveform"):
            ram.run(simple_env, source, receiver,
                    run_mode=RunMode.TIME_SERIES)

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_ram_compute_time_series_helper(self, simple_env, source):
        """Verify the convenience method ``RAM.compute_time_series`` runs.

        The helper takes no ``frequencies=`` so the auto-derive in
        ``_band.time_series_band`` fires; the warning is
        expected behaviour, filtered here.
        """
        from uacpy.core.results import Field
        ram = RAM(q_factor=2.0, record_duration=2.0, dr=20.0, dz=2.0, verbose=False)
        receiver = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
        fs = 4000.0
        nt = 64
        t = np.arange(nt) / fs
        sigma = nt / (8.0 * fs)
        f0 = float(np.atleast_1d(source.frequencies)[0])
        wf = (np.sin(2 * np.pi * f0 * (t - t[-1] / 2))
              * np.exp(-((t - t[-1] / 2) ** 2) / (2 * sigma ** 2)))
        result = ram.compute_time_series(
            simple_env, source, receiver,
            source_waveform=wf, sample_rate=fs,
        )
        assert isinstance(result, Field)
        assert result.data.shape[0] == 1
        assert result.data.shape[1] == 1

    def test_compute_time_series_forwards_output_duration(self, simple_env, source):
        """compute_time_series must forward output_duration (+ waveform/rate)
        to run() — it's the knob that sets the synthesised animation window."""
        receiver = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
        ram = RAM(verbose=False)
        captured = {}

        def _spy(env, src, rcv, *, run_mode=None, **kw):
            captured.update(run_mode=run_mode, **kw)
            return object()

        ram.run = _spy
        wf = np.ones(8)
        ram.compute_time_series(simple_env, source, receiver,
                                source_waveform=wf, sample_rate=4000.0,
                                output_duration=0.5)
        assert captured['run_mode'] is RunMode.TIME_SERIES
        assert captured['output_duration'] == 0.5
        assert captured['sample_rate'] == 4000.0
        assert captured['source_waveform'] is wf
        assert captured['frequencies'] is None

    def test_compute_time_series_forwards_frequencies(self, simple_env, source):
        """The auto-grid warning names ``frequencies=`` as the remedy; the
        wrapper the user called has to accept it."""
        receiver = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
        ram = RAM(verbose=False)
        captured = {}

        def _spy(env, src, rcv, *, run_mode=None, **kw):
            captured.update(run_mode=run_mode, **kw)
            return object()

        ram.run = _spy
        grid = np.arange(100.0, 301.0)
        ram.compute_time_series(simple_env, source, receiver,
                                source_waveform=np.ones(8), sample_rate=4000.0,
                                frequencies=grid)
        assert captured['frequencies'] is grid

    @pytest.mark.parametrize('bad', [float('nan'), 0.0, -1.0])
    def test_a_non_positive_output_duration_is_refused(self, simple_env,
                                                       source, bad):
        receiver = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
        with pytest.raises(ConfigurationError, match="output_duration"):
            RAM(verbose=False).run(simple_env, source, receiver,
                                   run_mode=RunMode.TIME_SERIES,
                                   source_waveform=np.ones(64),
                                   sample_rate=1000.0, output_duration=bad)


class TestNearRangePathsSteeperThanTheBand:
    """Source 25 m, receiver 60 m: the surface-reflected path spans 85 m,
    so a 30° band is clear of it from 85 / tan 30° = 147.2 m on, and a
    2400 m/s rock's 51.3° critical angle from 68.1 m on."""

    def _notices(self, bottom, closest_range):
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0, bottom=bottom)
        with recorded_warnings() as record:
            RAM().run_settings(
                env, uacpy.Source(depths=25.0, frequencies=200.0),
                Receiver(depths=60.0, ranges=[closest_range, 1000.0]))
        return [str(w.message) for w in record
                if w.category is NumericsWarning
                and 'angular band' in str(w.message)]

    def test_a_receiver_inside_the_clear_range_warns_and_names_it(self):
        (message,) = self._notices('sand', 147.0)
        assert 'band reaches 30.0° (angle_max=30°)' in message
        assert 'from 147 m on' in message
        assert 'Scooter(c_high=1e9)' in message

    def test_a_receiver_past_the_clear_range_is_silent(self):
        assert self._notices('sand', 147.5) == []

    def test_a_fast_seabed_widens_the_band_to_its_critical_angle(self):
        rock = BoundaryProperties(sound_speed=2400.0, density=2.5,
                                  attenuation=0.1)
        assert self._notices(rock, 68.5) == []
        (message,) = self._notices(rock, 67.5)
        assert 'band reaches 51.3° (the critical angle' in message



class TestTheAutomaticGridRunsAtKilohertz:
    """The Δz ladder reaches λ/200 above 750 Hz (``dz_ladder_end``) and the
    depth budget is 100 000 points, so the default grid converges at kHz
    where the 0.01 m ladder end refused every grid."""

    def test_5_khz_on_the_default_grid_agrees_with_kraken(self):
        from uacpy.models import Kraken
        env = uacpy.Environment(bathymetry=100.0, bottom='sand',
                                absorption=Thorp())
        source = uacpy.Source(depths=25.0, frequencies=5000.0)
        receiver = Receiver(depths=np.linspace(5.0, 95.0, 19),
                            ranges=np.linspace(500.0, 2000.0, 16))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ram = RAM().run(env, source, receiver).to_dB().data
            kraken = Kraken().run(env, source, receiver).to_dB().data
        diff = np.abs(ram - kraken)
        diff = diff[np.isfinite(diff)]
        # Measured 0.53 dB median, 1.22 dB rms (dz = 4.6 mm, 23 117 points).
        assert np.median(diff) < 1.0
        assert np.sqrt(np.mean(diff ** 2)) < 2.0



class TestTheSeabedPadHoldsTheContinuousSpectrum:
    """``leaky_field_depth``: the depth a transmitted component's bottom field
    reaches (``r·tan θ_b``) while its leakage or the seabed's loss keeps it
    within 20 dB, and never past the farthest receiver."""

    @staticmethod
    def _depth(bottom, max_range, freq=100.0, depth=240.0):
        env = Environment(bathymetry=depth, ssp=1500.0, bottom=bottom)
        return ram_domain.leaky_field_depth(
            env, freq, max_range, knobs=RAM(verbose=False)._knob_record())

    def test_a_dense_lossless_bottom_near_the_water_speed_needs_a_deep_pad(
            self):
        bucker = BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1505.0, density=2.1,
                                    attenuation=0.0)
        # 20 km: about 1.1 km (zmax 1659 m on Bucker's own profile); the
        # reach grows with the track and stops at its end.
        far, near = self._depth(bucker, 20000.0), self._depth(bucker, 2000.0)
        assert 900.0 < far < 1400.0
        assert near < far

    def test_a_lossy_bottom_bounds_the_pad_by_its_own_loss(self):
        lossy = BoundaryProperties(acoustic_type='half-space',
                                   sound_speed=1505.0, density=2.1,
                                   attenuation=0.5)
        # 20 dB over z / sin θ_b at 0.5 dB/λ_b: z <= 40 λ_b sin θ_b.
        assert self._depth(lossy, 20000.0) <= 40.0 * 15.05

    def test_a_vacuum_seabed_has_no_field_below_it(self):
        assert self._depth(BoundaryProperties(acoustic_type='vacuum'),
                           20000.0) == 0.0

    _SOFT = BoundaryProperties(acoustic_type='half-space', sound_speed=1500.0,
                               density=1e-4, attenuation=0.0)

    def test_a_near_perfect_reflector_needs_a_pad_once_it_has_leaked_1_percent(
            self):
        # |R| = 0.9998 loses 2.5e-6 dB/m at 30°: 1 % of its energy (0.0436
        # dB) is gone after about 17.4 km, and only then is a pad needed.
        assert self._depth(self._SOFT, 17000.0, freq=25.0, depth=200.0) == 0.0
        assert self._depth(self._SOFT, 18000.0, freq=25.0, depth=200.0) > 0.0

    def test_the_aperture_is_the_sources_and_not_the_reliefs(self):
        # A 4 m step written as a 1 m ramp reads as a 76° slope; the relief
        # widens the Padé bracket and leaves the pad alone. ``angle_max``
        # at 76° leaks 1 % within 4 km over the same seabed.
        step = Environment(
            bathymetry=Bathymetry(ranges=np.array([0.0, 1499.5, 1500.5, 6000.0]),
                                  depths=np.array([200.0, 200.0, 204.0, 204.0])),
            ssp=1500.0, bottom=self._SOFT)
        flat = Environment(bathymetry=204.0, ssp=1500.0, bottom=self._SOFT)
        assert ram_domain.leaky_field_depth(
            step, 25.0, 4000.0, knobs=RAM(verbose=False)._knob_record()) == 0.0
        assert ram_domain.leaky_field_depth(
            flat, 25.0, 4000.0,
            knobs=RAM(verbose=False, angle_max=76.0)._knob_record()) > 0.0


class TestAFrancoisGarrisonProfileInTheWaterBlock:
    """A Francois-Garrison T/S profile reaches RAM's water block as the
    formula evaluated with each depth's own water, its rows sampled; a
    uniform profile is the one-row law."""

    F = 4000.0

    def test_one_water_row_is_the_alpha_the_at_deck_rows_carry(
            self, tmp_path):
        """One Francois-Garrison water row: RAM's water block and a Kraken
        deck's ``alphaI`` rows hold the same α(z) — the formula at each
        depth, in dB per local wavelength — at every SSP node, to the
        deck's nine printed digits; the depth term is live on both."""
        from uacpy.io.oalib_writer import write_kraken_env_file
        from uacpy.tests.conftest import at_deck_water_rows
        env = Environment(name='deep', bathymetry=4000.0,
                          ssp=[(0.0, 1500.0), (1000.0, 1480.0),
                               (4000.0, 1530.0)],
                          bottom='sand',
                          absorption=FrancoisGarrison(4.0, 34.5, 7.9))
        write_kraken_env_file(tmp_path / 'k.env', env,
                              Source(depths=30.0, frequencies=self.F),
                              Receiver(depths=[20.0], ranges=[1000.0]),
                              c_low=1400.0, c_high=2000.0)
        rows = at_deck_water_rows((tmp_path / 'k.env').read_text())
        assert rows[:, 0].tolist() == [0.0, 1000.0, 4000.0]
        m = RAM(verbose=False, earth_curvature=False)
        block = dict(ram_collins.collins_range_segments(
            env, 'ramgeo', 4500.0, self.F, dz=0.5, knobs=m._knob_record(),
            speed_bounds=m._speed_bounds)[0]['water_attn'])
        for z, _c, alpha in rows:
            assert block[z] == pytest.approx(alpha, rel=5e-9), z
        per_metre = rows[:, 2] / rows[:, 1]              # alpha / f, dB/m/Hz
        assert abs(per_metre[2] / per_metre[0] - 1.0) > 0.05

    def test_the_block_holds_the_formula_at_each_depth(self):
        from uacpy.tests.conftest import (
            TWO_LAYER_DEPTHS, two_layer_absorption, two_layer_dB_per_m)
        env = Environment(name='two-layer', bathymetry=100.0, ssp=1500.0,
                          bottom='sand', absorption=two_layer_absorption())
        m = RAM(verbose=False, earth_curvature=False)
        block = ram_collins.collins_range_segments(
            env, 'ramgeo', 200.0, self.F, dz=0.05, knobs=m._knob_record(),
            speed_bounds=m._speed_bounds)[0]['water_attn']
        depths = np.array([z for z, _ in block])
        # The profile's rows are sampling points of their own.
        assert set(TWO_LAYER_DEPTHS) <= set(depths)
        for z, value in block:
            if z > 100.0:
                continue
            expected = two_layer_dB_per_m(self.F, z) * 1500.0 / self.F
            assert value == pytest.approx(expected, rel=1e-12), z

    def test_a_uniform_profile_gives_the_one_row_field_to_the_bit(self):
        # Profile rows on the grid's own points (0 m and the seafloor) add
        # no sample, so RAM sees the same alpha at the same depths.
        row = FrancoisGarrison(12.0, 34.5, 8.0)
        column = FrancoisGarrison([12.0, 12.0], [34.5, 34.5], 8.0,
                                  depths=[0.0, 100.0])
        fields = [np.asarray(RAM(verbose=False).run(
            Environment(name='fg', bathymetry=100.0, ssp=1500.0,
                        bottom='sand', absorption=law),
            Source(depths=30.0, frequencies=self.F),
            Receiver(depths=[20.0, 60.0],
                     ranges=np.array([500.0, 1500.0, 3000.0])),
            run_mode=RunMode.COHERENT_TL).data) for law in (row, column)]
        np.testing.assert_array_equal(fields[1], fields[0])
