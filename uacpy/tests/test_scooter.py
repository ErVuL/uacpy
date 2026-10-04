"""Scooter wavenumber-integration-focused tests."""

import warnings

import pytest
import numpy as np

from uacpy.core.results import Field
from uacpy.models import Scooter
from uacpy.models.scooter._settings import ScooterSettings
from uacpy.models.scooter._extract import taper_bounds
from uacpy.models._budget import UNREADABLE_HOST_BYTES
from uacpy.models.scooter._plan import (
    _TRANSFORM_BYTES_PER_ELEMENT, mesh_floor_notice,
    reject_oversized_green_cube,
)
from uacpy.core.run_settings import RunMode
from uacpy.core import Environment, Source, Receiver
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.tests.conftest import make_pekeris
from uacpy.tests.conftest import make_halfspace
from uacpy.tests.conftest import recorded_warnings

pytestmark = pytest.mark.requires_binary


class TestScooterBasic:
    """Basic tests for Scooter model (wavenumber integration)."""

    @pytest.mark.requires_binary
    def test_compute_tl_returns_finite_grid_matching_receiver_shape(self):
        """Test basic Scooter TL computation."""
        env = Environment(
            name="scooter_test",
            bathymetry=100.0,
            ssp=1500.0
        )
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0])
        )

        scooter = Scooter(verbose=False)
        result = scooter.compute_tl(env=env, source=source, receiver=receiver)

        assert isinstance(result, Field)
        assert result.shape == (len(receiver.depths), len(receiver.ranges))
        assert np.all(np.isfinite(result.data))


class TestScooterBroadband:
    """End-to-end BROADBAND / TIME_SERIES tests for Scooter."""

    @pytest.mark.slow
    def test_scooter_broadband_returns_transfer_function(self):
        """Scooter BROADBAND returns a populated H(f) Field."""
        env = Environment(name="sc_bb", bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0]),
        )
        frequencies = np.linspace(80.0, 120.0, 5)

        scooter = Scooter(verbose=False)
        result = scooter.run(
            env, source, receiver,
            run_mode=RunMode.BROADBAND,
            frequencies=frequencies,
        )

        assert isinstance(result, Field)
        assert np.iscomplexobj(result.data)
        assert result.data.shape[:2] == (len(receiver.depths), len(receiver.ranges))
        assert result.data.shape[2] > 0
        assert np.all(np.isfinite(result.data))
        assert np.any(np.abs(result.data) > 0)

        # The 100 Hz slice of H(f) is the same physics as a COHERENT_TL run of
        # the same environment (pattern: test_ram_backends.py
        # ``test_broadband_fc_slice_matches_the_narrowband_run``). The two runs
        # sample different wavenumber grids (rmax_factor 3.0 vs 2.0), so
        # the comparison bounds the median bias rather than pinning equality.
        nb = Scooter(verbose=False).compute_tl(env=env, source=source,
                                               receiver=receiver)
        got = np.asarray(result.at(frequency=100.0).to_dB().data, dtype=float)
        ref = np.asarray(nb.dB, dtype=float)
        ok = np.isfinite(got) & np.isfinite(ref)
        assert ok.any()
        bias = float(np.median(got[ok] - ref[ok]))
        assert abs(bias) < 0.5, (
            f"BROADBAND 100 Hz slice sits {bias:+.3f} dB off the COHERENT_TL "
            f"run of the same environment")

    @pytest.mark.slow
    def test_scooter_time_series_returns_time_series_field(self):
        """Scooter TIME_SERIES with a tonal waveform returns Field."""
        env = Environment(name="sc_ts", bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=np.array([50.0]),
            ranges=np.array([2000.0]),
        )
        fs = 2000.0
        n = 256
        t = np.arange(n) / fs
        waveform = np.sin(2 * np.pi * 100.0 * t) * np.hanning(n)
        # Δf small enough that 1/Δf ≥ waveform duration (256/2000 = 0.128s)
        # → no DFT-wraparound warning from synthesize_time_series.
        frequencies = np.linspace(60.0, 140.0, 17)

        scooter = Scooter(verbose=False)
        result = scooter.run(
            env, source, receiver,
            run_mode=RunMode.TIME_SERIES,
            frequencies=frequencies,
            source_waveform=waveform,
            sample_rate=fs,
        )

        assert isinstance(result, Field)
        assert result.data.shape[0] == len(receiver.depths)
        assert result.data.shape[1] == len(receiver.ranges)
        assert result.data.shape[2] > 0
        assert np.all(np.isfinite(result.data))

    def test_scooter_time_series_requires_waveform(self):
        """Scooter TIME_SERIES without source_waveform must raise."""
        env = Environment(name="sc_ts_err", bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=np.array([50.0]),
            ranges=np.array([2000.0]),
        )
        scooter = Scooter(verbose=False)
        with pytest.raises(ConfigurationError, match="source_waveform"):
            scooter.run(
                env, source, receiver,
                run_mode=RunMode.TIME_SERIES,
            )


def test_scooter_constructor_rejects_source_type():
    """Source geometry belongs to the ``Source`` carrier, not the solver:
    ``scooter._extract.assemble_field_from_grn`` reads ``source.source_type``. A
    duplicate on the model would let the deck and the carrier disagree about
    what was radiated.
    """
    with pytest.raises(TypeError,
                       match="unexpected keyword argument 'source_type'"):
        Scooter(source_type='R')


def test_scooter_constructor_rejects_field_interp():
    """``field_interp`` named an FLP option for ``fields.exe``, which uacpy
    never runs — the k→r transform is done in-tree."""
    with pytest.raises(TypeError,
                       match="unexpected keyword argument 'field_interp'"):
        Scooter(field_interp='P')


def test_grn_transform_is_recorded_as_the_direct_dft():
    """The transform is a trapezoidal-rule DFT (``fieldsco.m:5``), not an FFT,
    and the field says so."""
    from uacpy.core.results import GreensFunction
    gf = GreensFunction(data=np.ones((1, 1, 1, 8), np.complex64),
                        phase_speeds=np.linspace(1700.0, 1400.0, 8),
                        receiver_depths=[10.0], source_depths=[5.0],
                        frequencies=100.0, title='SCOOTER - test')
    field = gf.to_field(np.array([100.0]))
    assert field.metadata['transform_method'] == 'direct_dft'


class TestZeroReceiverRange:
    """A receiver at ``r = 0`` sits on the point source's cylindrical-spreading
    singularity (``1/sqrt(r)``). ``fieldsco.m:69`` sidesteps it by moving the
    range to 1 m; uacpy reports no-data instead, and every model must report
    the same thing on the same grid."""

    RANGES = np.array([0.0, 1000.0, 3000.0])

    @staticmethod
    def _env():
        return Environment(name='zero_r', bathymetry=100.0, ssp=1500.0)

    @staticmethod
    def _source():
        return Source(depths=50.0, frequencies=100.0)

    def _receiver(self):
        return Receiver(depths=np.array([25.0, 75.0]), ranges=self.RANGES)

    def test_scooter_zero_range_is_no_data_not_a_huge_number(self):
        receiver = self._receiver()
        with pytest.warns(UserWarning, match="r = 0"):
            result = Scooter(verbose=False).run(
                self._env(), self._source(), receiver)
        data = np.asarray(result.data)
        assert np.all(np.isnan(data[:, 0]))
        assert np.all(np.isfinite(data[:, 1:]))
        # A clamped denominator turns the singular cell into |p| ~ 1e152
        # (TL ~ -3000 dB), which poisons every colour scale and every max/mean
        # over the grid. Beyond a wavelength or so from a unit-amplitude point
        # source |p| < 1 everywhere, so 1.0 separates a physical field from a
        # blown-up one by ~150 orders of magnitude.
        assert np.nanmax(np.abs(data)) < 1.0

    def test_kraken_and_scooter_agree_on_the_zero_range_cell(self):
        from uacpy.models import Kraken

        env, source = self._env(), self._source()
        with pytest.warns(UserWarning, match="r = 0"):
            scooter_tl = np.asarray(
                Scooter(verbose=False).run(env, source, self._receiver()).dB)
        with pytest.warns(UserWarning, match="r = 0"):
            kraken_tl = np.asarray(
                Kraken(verbose=False).compute_tl(
                    env, source, self._receiver()).dB)

        assert np.all(np.isnan(scooter_tl[:, 0]))
        assert np.all(np.isnan(kraken_tl[:, 0]))
        assert np.all(np.isfinite(scooter_tl[:, 1:]))
        assert np.all(np.isfinite(kraken_tl[:, 1:]))


def test_a_receiver_with_no_positive_range_is_refused_before_launch():
    """Scooter's spectral RMax derives from the maximum receiver range
    (``RMax = range_max × rmax_factor``), so a receiver whose ranges
    default to the single point at 0 m is refused with
    ``ConfigurationError`` instead of reaching the binary's unexplained
    STOP."""
    env = Environment(name='no_range', bathymetry=100.0, ssp=1500.0)
    source = Source(depths=50.0, frequencies=100.0)
    with pytest.warns(UserWarning, match='ranges not given'):
        receiver = Receiver(depths=np.array([25.0, 75.0]))
    with pytest.raises(ConfigurationError, match='positive receiver range'):
        Scooter(verbose=False).run(env, source, receiver)


class TestScooterReceiverDepthAxis:
    """The settled below-domain policy (``PropagationModel.validate_inputs``):
    receivers are outputs, so a receiver below the deepest modelled interface
    is accepted and returns the model's below-domain value. The returned depth
    axis must therefore be the one the caller asked for — moving receivers onto
    the mesh and de-duplicating them silently misaligns every row a caller
    indexes against its own depth array."""

    @staticmethod
    def _env():
        from uacpy.core import BoundaryProperties
        from uacpy.core.boundary import SedimentLayer
        from uacpy.core.bottom import Bottom, SeabedColumn
        column = SeabedColumn(
            layers=[SedimentLayer(thickness=20.0, sound_speed=1600.0,
                                  density=1.8, attenuation=0.5)],
            halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=2.0, attenuation=0.8))
        return Environment(name='media', bathymetry=200.0, ssp=1500.0,
                           bottom=Bottom(columns=[column]))

    def test_requested_depths_are_returned_verbatim(self):
        # 200 m water + 20 m sediment ⇒ resolvable to 220 m; 300 and 400 sit
        # below it and must not collapse onto one 217 m row.
        receiver = Receiver(depths=np.array([50.0, 150.0, 300.0, 400.0]),
                            ranges=np.linspace(500.0, 5000.0, 5))
        result = Scooter(verbose=False).run(
            self._env(), Source(depths=50.0, frequencies=100.0), receiver)
        assert result.data.shape[0] == receiver.depths.size
        assert np.asarray(result.coords['depth']) == pytest.approx(
            receiver.depths)

    def test_unresolvable_depths_are_no_data(self):
        """Below the mesh the binary clamps onto the deepest interface; that
        value belongs to a different depth, so it must not be handed back."""
        receiver = Receiver(depths=np.array([50.0, 150.0, 300.0, 400.0]),
                            ranges=np.linspace(500.0, 5000.0, 5))
        data = np.asarray(Scooter(verbose=False).run(
            self._env(), Source(depths=50.0, frequencies=100.0), receiver).data)
        assert np.all(np.isfinite(data[:2]))
        assert np.all(np.isnan(data[2:]))


class TestScooterRejectsATooCoarseMesh:
    """SCOOTER reads its deck through the same ``misc/ReadEnvironmentMod.f90``
    as KRAKEN, so a pinned ``n_mesh`` under the ``Nneeded / 2`` floor at
    ``:110-112`` stops the binary with *Mesh is too coarse*. The guard has to
    be on Scooter too, or the condition reaches the caller as a bare Fortran
    fatal instead of a ConfigurationError."""

    @staticmethod
    def _env():
        from uacpy.core import BoundaryProperties
        return Environment(
            name='pekeris', bathymetry=200.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.3))

    def test_a_too_coarse_n_mesh_is_a_configuration_error(self):
        with pytest.raises(ConfigurationError, match='Mesh is too coarse'):
            Scooter(n_mesh=5, verbose=False).run(
                self._env(), Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[100.0], ranges=np.linspace(1000.0, 5000.0, 5)))

    def test_auto_mesh_is_never_rejected(self):
        """``NG = 0`` is AT's automatic sizing and is never range-checked."""
        result = Scooter(verbose=False).run(
            self._env(), Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[100.0], ranges=np.linspace(1000.0, 5000.0, 5)))
        assert np.all(np.isfinite(np.asarray(result.data)))


class TestStabilisingAttenuationIsUndoneCorrectly:
    """``scooter.f90:581`` evaluates the FE solve on the contour ``k + i*Atten``,
    so the inverse transform must undo that offset with ``exp(+Atten*r)`` using
    the ``Atten`` the solver actually used.

    ``scooter.f90:122-125`` recomputes ``Deltak`` inside the frequency loop from
    ``kMin = omega/cHigh``, so ``Atten`` scales with frequency while
    ``scooter.f90:133`` writes the header only for ``ifreq == 1``. That is why
    ``Matlab/Scooter/fieldsco.m:113-115`` re-derives it from the file's own ``k``
    vector, and why that is right for a broadband run. But ``scooter.f90:130``
    zeroes ``Atten`` at *every* frequency when ``TopOpt(7:7) == '0'``, so a zero
    header is valid throughout — and re-deriving ``Δk`` there multiplies a
    real-axis Green's function by ``exp(+Δk*r)``.
    """

    @staticmethod
    def _grn(atten, k, title='SCOOTER - test', times=None):
        from uacpy.core.results import GreensFunction
        n = 1 if times is None else len(times)
        return GreensFunction(
            data=np.zeros((n, 1, 1, 1), np.complex64),
            phase_speeds=[1500.0], receiver_depths=[10.0],
            source_depths=[5.0], frequencies=100.0, times=times,
            stabilizing_attenuation=atten, title=title), k

    def test_a_zero_header_is_honoured(self):
        """The solver wrote 0 because TopOpt(7:7) = '0'; Δk must not come back."""
        grn, k = self._grn(0.0, np.array([1.0, 1.0 + 1.5745e-4, 1.0 + 3.149e-4]))
        assert grn._contour_attenuation(k) == 0.0

    def test_a_non_zero_header_is_re_derived_per_frequency(self):
        """With the stabiliser on, the header carries only the first frequency's
        Δk, so the k vector is the authority."""
        dk = 1.5745e-4
        grn, k = self._grn(9.9e-9, np.array([1.0, 1.0 + dk, 1.0 + 2 * dk]))
        assert grn._contour_attenuation(k) == pytest.approx(dk)

    def test_another_title_keeps_its_header(self):
        grn, k = self._grn(9.9e-9, np.array([1.0, 2.0]), title='OTHER')
        assert grn._contour_attenuation(k) == 9.9e-9

    def test_sparc_is_unaffected(self):
        grn, k = self._grn(1.0, np.array([1.0, 2.0]), title='SPARC- snap',
                           times=[0.0, 0.1])
        assert grn._contour_attenuation(k) == 0.0

    def test_turning_the_stabiliser_off_warns_with_its_cost(self):
        """Removing the contour offset puts the modal poles back on the
        integration path, which a correct transform cannot repair."""
        with pytest.warns(UserWarning, match='modal poles'):
            Scooter(stabilizing_attenuation_off=True)
        with recorded_warnings() as caught:
            Scooter()
        assert not [c for c in caught if 'modal poles' in str(c.message)]


def test_broadband_n_mesh_is_checked_at_the_deck_freq0():
    """A pinned ``n_mesh`` is checked against the AT reader's floor at the
    deck's ``freq0`` — the first frequency of the sweep — because that is
    the only place the reader applies it.

    ``misc/ReadEnvironmentMod.f90:103-112`` sizes the requirement from
    ``freq0`` during the environment read; ``Scooter/scooter.f90:106`` then
    marches each swept frequency on a mesh scaled by ``freq/freq0``. A mesh
    clearing the floor at ``freq0`` therefore clears it for the whole
    sweep, so validating at the top of the band would refuse decks the
    binary runs.
    """
    import warnings as _w
    env = Environment(bathymetry=100.0, ssp=1500.0)
    sweep = Source(depths=25.0, frequencies=np.linspace(100.0, 1000.0, 10))
    rcv = Receiver(depths=[50.0], ranges=[1000.0])

    # Clears the floor at 100 Hz, far under it at 1000 Hz: accepted.
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        Scooter(n_mesh=200, verbose=False).run(
            env, sweep, rcv, run_mode=RunMode.BROADBAND)

    # Below the floor at freq0 itself: still refused.
    with pytest.raises(ConfigurationError, match='Mesh is too coarse'):
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            Scooter(n_mesh=3, verbose=False).run(
                env, sweep, rcv, run_mode=RunMode.BROADBAND)


class TestScooterDeckResolution:
    """Deck-level checks of the documented ``None`` resolutions
    (``docs/models/scooter.md`` constructor table): the spectral
    ``RMax = receiver.ranges.max() × rmax_factor`` with the multiplier
    defaulting to 2.0 narrowband / 3.0 broadband, and
    ``c_low = 0.95 × min(SSP)`` when only ``c_high`` is pinned. Read off
    ``run_settings(...).engine``, the settings the deck is written from,
    without launching the binary (``TestTheDeckIsWrittenFromTheResolvedSettings``
    pins that the deck carries them)."""

    @staticmethod
    def _env():
        from uacpy.core import BoundaryProperties
        return Environment(
            name='deck', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5))

    @staticmethod
    def _engine(model, run_mode=RunMode.COHERENT_TL, frequencies=None):
        return model.run_settings(
            TestScooterDeckResolution._env(),
            Source(depths=50.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]),
                     ranges=np.array([1000.0, 3000.0])),
            run_mode, frequencies=frequencies).engine

    def test_narrowband_rmax_is_twice_the_receiver_max(self):
        # 0.95×1500 = 1425.0, 1.05×1800 = 1890.0; RMax = 3000 m × 2.0.
        engine = self._engine(Scooter(verbose=False))
        assert (engine.c_low, engine.c_high) == pytest.approx((1425.0, 1890.0))
        assert engine.rmax_m == pytest.approx(6000.0)
        assert engine.rmax_factor_origin == 'the COHERENT_TL default'

    def test_broadband_rmax_is_three_times_the_receiver_max(self):
        engine = self._engine(
            Scooter(verbose=False), run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(80.0, 120.0, 5))
        assert engine.rmax_m == pytest.approx(9000.0)
        assert engine.rmax_factor == 3.0

    def test_pinned_rmax_factor_wins_in_both_modes(self):
        model = Scooter(rmax_factor=5.0, verbose=False)
        narrow = self._engine(model)
        broad = self._engine(model, run_mode=RunMode.BROADBAND,
                             frequencies=np.linspace(80.0, 120.0, 5))
        assert narrow.rmax_m == pytest.approx(15000.0)
        assert broad.rmax_m == pytest.approx(15000.0)
        assert broad.rmax_factor_origin == 'Scooter(rmax_factor=…)'

    def test_c_low_auto_derives_with_c_high_pinned(self):
        # Only c_high pinned: c_low still resolves to 0.95 × min(SSP).
        engine = self._engine(Scooter(c_high=1700.0, verbose=False))
        assert (engine.c_low, engine.c_high) == pytest.approx((1425.0, 1700.0))
        assert engine.c_low_origin == '0.95 × min(env.ssp)'
        assert engine.c_high_origin == 'Scooter(c_high=…)'

    def test_documented_factors_are_the_code_factors(self):
        # scooter.md and DOCUMENTATION.md state 0.95 / 1.05 as literals; this
        # pins the constants so doc-vs-code drift fails a test.
        from uacpy.models._window import C_LOW_FACTOR, C_HIGH_FACTOR
        assert C_LOW_FACTOR == 0.95
        assert C_HIGH_FACTOR == 1.05


class TestScooterSpectrumOption:
    """``wavenumber_spectrum`` names the wavenumber branch the k→r transform integrates
    (``docs/models/scooter.md`` constructor table), in the words
    ``hankel_transform`` takes as ``spectrum``."""

    @pytest.mark.parametrize('name', ['positive', 'negative', 'both'])
    def test_spectrum_name_is_a_transform_word(self, name):
        from uacpy.core.acoustics import hankel_transform
        assert Scooter(wavenumber_spectrum=name,
                       verbose=False).wavenumber_spectrum == name
        p = hankel_transform(np.ones((1, 8), dtype=complex),
                             np.linspace(0.1, 0.8, 8), np.array([10.0]),
                             attenuation=0.0, spectrum=name)
        assert p.shape == (1, 1)

    def test_unknown_spectrum_raises(self):
        with pytest.raises(ConfigurationError, match='wavenumber_spectrum'):
            Scooter(wavenumber_spectrum='full', verbose=False)

    def test_the_bare_spectrum_keyword_is_not_accepted(self):
        with pytest.raises(TypeError, match='spectrum'):
            Scooter(spectrum='positive', verbose=False)


class TestNMeshSilentFloor:
    """``scooter.f90:110`` floors ``n_mesh`` at 100 points per medium without
    any echo of the override, so sub-100 values change nothing
    (``docs/models/scooter.md`` §7): 40 and 100 give bit-identical TL and only
    a value above the floor moves the answer."""

    @staticmethod
    def _tl(n_mesh):
        from uacpy.core import BoundaryProperties
        env = Environment(
            name='floor', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5))
        result = Scooter(n_mesh=n_mesh, verbose=False).compute_tl(
            env=env, source=Source(depths=50.0, frequencies=50.0),
            receiver=Receiver(depths=np.array([25.0, 75.0]),
                              ranges=np.linspace(500.0, 3000.0, 6)))
        return np.asarray(result.dB, dtype=float)

    @pytest.mark.requires_binary
    def test_sub_floor_n_mesh_is_bit_identical_to_the_floor(self):
        assert np.array_equal(self._tl(40), self._tl(100))

    @pytest.mark.requires_binary
    def test_run_announces_the_sub_floor_n_mesh(self):
        with pytest.warns(UserWarning, match='has no effect'):
            self._tl(40)

    @pytest.mark.requires_binary
    def test_above_floor_n_mesh_moves_the_answer(self):
        assert not np.array_equal(self._tl(150), self._tl(100))

    @staticmethod
    def _floor_warnings(n_mesh, freqs, freq0=50.0):
        notice = mesh_floor_notice(
            freq0, freqs,
            n_mesh=Scooter(n_mesh=n_mesh, verbose=False).n_mesh)
        return [] if notice is None else [notice[1]]

    @pytest.mark.parametrize('n_mesh, warned', [
        (99, True),    # INT(99) < 100: the binary runs 100
        (100, False),  # the floor itself is what runs
        (0, False),    # auto mesh: nothing pinned to override
    ])
    def test_a_pinned_n_mesh_under_the_floor_warns(self, n_mesh, warned):
        assert bool(self._floor_warnings(n_mesh, [50.0])) is warned

    def test_the_floor_is_judged_per_frequency_of_the_sweep(self):
        # N scales with freq/freq0: 150 at 50 Hz, 60 at 20 Hz.
        said = self._floor_warnings(150, [20.0, 50.0, 100.0])
        assert len(said) == 1 and 'at 20 Hz' in said[0], said
        assert not self._floor_warnings(150, [50.0, 100.0])


class TestPrecalcBottomIrcGuard:
    """A 'precalc' bottom stages the user's file verbatim as ``<base>.irc``,
    which ``misc/RefCoef.f90:94-107`` reads as Title/freq + NkTab +
    ``(5G15.7,I5)`` f/g-impedance records — a different format from the
    ``.brc``/``.trc`` angle tables. The natural mistake (handing it a
    theta/|R|/phase table) used to abort the binary with a bare Fortran
    backtrace at exit 2; the header is validated before launch instead."""

    @staticmethod
    def _run(table):
        from uacpy.core import BoundaryProperties
        env = Environment(
            name='precalc', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='precalc',
                                      reflection_file=str(table)))
        return Scooter(verbose=False).run(
            env, Source(depths=25.0, frequencies=50.0),
            Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])))

    def test_angle_table_raises_typed_error_before_launch(self, tmp_path):
        table = tmp_path / 'angles.brc'
        table.write_text("3\n0.0 1.0 0.0\n45.0 0.5 0.0\n90.0 0.0 0.0\n")
        with pytest.raises(ConfigurationError,
                           match='line 1 is all-numeric') as err:
            self._run(table)
        msg = str(err.value)
        assert '.irc' in msg and '.brc' in msg
        assert "acoustic_type='file'" in msg

    def test_irc_shaped_header_passes_the_guard(self, tmp_path):
        # BOUNCE's layout: quoted-title + freq, NkTab, (5G15.7,I5) records.
        table = tmp_path / 'seabed.irc'
        table.write_text(
            "'seabed' 50.0\n2\n"
            "  0.1000000E+00  0.2000000E+00  0.0000000E+00  0.3000000E+00"
            "  0.0000000E+00    0\n"
            "  0.2000000E+00  0.2500000E+00  0.0000000E+00  0.3500000E+00"
            "  0.0000000E+00    0\n")
        from uacpy.core import BoundaryProperties
        env = Environment(
            name='precalc', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='precalc',
                                      reflection_file=str(table)))
        # The guard alone: header accepted, no exception raised.
        assert Scooter(verbose=False)._reject_malformed_irc_bottom(env) is None


@pytest.mark.requires_binary
class TestScooterBroadbandStampsThePhysicalCMax:
    """A 3000 m/s half-space under 1500 m/s water — the configuration where
    an unstamped result anchored ``to_time_trace`` at 1500 m/s and the
    early bottom-refracted arrivals wrapped."""

    def test_the_stamp_is_the_seabed_speed_and_anchors_the_window(self):
        env = Environment(name='cmax_bb', bathymetry=100.0, ssp=1500.0,
                          bottom=make_halfspace(3000.0, density=2.0,
                                            attenuation=0.1))
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([50.0]), ranges=np.array([2000.0]))
        result = Scooter(verbose=False).run(
            env, src, rcv, run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(80.0, 120.0, 5))

        assert result.run_settings.waveguide.c_max == pytest.approx(3000.0)
        assert 'c_max' not in result.metadata

        trace = result.to_time_trace(depth=50.0, range=2000.0)
        t = np.asarray(trace.coords['time'], dtype=float)
        # T_window = 1/df = 0.1 s, so the window opens half a window ahead
        # of the r / c_max = 0.667 s fastest possible arrival — not at the
        # 1.283 s a 1500 m/s default anchor gives.
        assert t[0] == pytest.approx(2000.0 / 3000.0 - 0.05, abs=0.02)


@pytest.mark.requires_binary
class TestScooterRefusesAPhaseSpeedBandInvertedByOnePinnedBound:
    """The constructor can only compare two pinned bounds. One pinned bound is
    comparable once the other is derived from the env, which happens when
    the run's settings are resolved; on the 100 m / 1500 m/s guide the auto
    band is (1425, 1680) m/s, so ``c_low=2000`` alone writes CLOW > CHIGH.
    ``ReadEnvironmentMod.f90:135`` then stops the binary after the deck has
    been written and the process spawned.
    """

    @staticmethod
    def _run(tmp_path, **kwargs):
        env = Environment(name='flat', bathymetry=100.0, ssp=1500.0)
        return Scooter(verbose=False, work_dir=tmp_path, cleanup=False,
                       **kwargs).run(
            env, Source(depths=25.0, frequencies=200.0),
            Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])))

    def test_a_pinned_c_low_above_the_derived_c_high_names_c_low(self, tmp_path):
        with pytest.raises(ConfigurationError, match='pinned c_low=2000'):
            self._run(tmp_path, c_low=2000.0)
        assert not (tmp_path / 'model.env').exists()

    def test_a_pinned_c_high_below_the_derived_c_low_names_c_high(self, tmp_path):
        with pytest.raises(ConfigurationError, match='pinned c_high=100'):
            self._run(tmp_path, c_high=100.0)

    def test_a_band_that_brackets_the_derived_bounds_writes_the_deck(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._run(tmp_path, c_low=1200.0)
        assert (tmp_path / 'model.env').exists()


class TestScooterRefusesAWavenumberGridItCannotSpace:
    """``scooter.f90:69`` derives ``Nk = INT(2000*RMax_km*(kMax-kMin)/pi)`` and
    has no test of the value — the only ``IF`` naming ``Nk`` is the allocation
    status at ``:74``. ``Nk = 1`` then divides by ``Nk - 1 = 0`` at ``:77`` and
    ``:125``, so the binary writes an all-NaN Green's function and exits 0;
    ``Nk = 0`` writes one with no samples at all. Both refused before launch,
    as ``models/bounce/`` and ``models/sparc/`` refuse the same arithmetic.

    On a 100 m isovelocity guide at 100 Hz with the band pinned to
    (1500, 1935) m/s, ``RMax = ranges.max() x 2`` puts the ``Nk = 1`` / ``2``
    boundary between a 16.6 m and a 16.7 m furthest receiver.
    """

    BAND = dict(c_low=1500.0, c_high=1935.0)

    @staticmethod
    def _call(tmp_path, r_max, entry='run_settings', **kwargs):
        env = Environment(name='flat', bathymetry=100.0, ssp=1500.0)
        model = Scooter(verbose=False, work_dir=tmp_path, cleanup=False,
                        **kwargs)
        return getattr(model, entry)(
            env, Source(depths=25.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]), ranges=np.array([r_max])))

    def _write(self, tmp_path, r_max, **kwargs):
        return self._call(tmp_path, r_max, entry='run', **kwargs)

    def test_a_single_sample_grid_is_refused_before_the_deck_is_written(
            self, tmp_path):
        with pytest.raises(ConfigurationError, match=r'Nk = 1 wavenumber'):
            self._write(tmp_path, 16.6, **self.BAND)
        assert not (tmp_path / 'model.env').exists()

    def test_two_samples_are_resolved(self, tmp_path):
        """The high side of the same boundary — 0.1 m further out."""
        assert self._call(tmp_path, 16.7, **self.BAND).engine.n_wavenumbers == 2

    def test_an_empty_grid_is_refused_before_the_deck_is_written(
            self, tmp_path):
        with pytest.raises(ConfigurationError, match=r'Nk = 0 wavenumber'):
            self._write(tmp_path, 8.3, **self.BAND)
        assert not (tmp_path / 'model.env').exists()

    def test_the_two_counts_are_refused_for_their_own_reasons(self, tmp_path):
        """``Nk = 1`` and ``Nk = 0`` fail differently in the Fortran, so the
        message must not describe one as the other."""
        with pytest.raises(ConfigurationError,
                           match='Nk = 1 wavenumber sample') as one:
            self._write(tmp_path, 16.6, **self.BAND)
        with pytest.raises(ConfigurationError,
                           match='Nk = 0 wavenumber sample') as zero:
            self._write(tmp_path, 8.3, **self.BAND)
        assert 'divides by zero' in str(one.value)
        assert 'divides by zero' not in str(zero.value)

    def test_the_remediation_names_the_knobs_that_raise_nk(self, tmp_path):
        """``Nk`` grows with ``RMax``, and ``RMax = ranges.max() x
        rmax_factor`` — so advice to shorten the receiver ranges lowers
        the count that is already too low."""
        with pytest.raises(ConfigurationError,
                           match='Nk = 1 wavenumber sample') as exc:
            self._write(tmp_path, 16.6, **self.BAND)
        text = str(exc.value)
        assert 'rmax_factor' in text
        assert 'c_low/c_high' in text
        assert 'shorten the receiver ranges' not in text

    def test_the_predicted_count_is_the_count_the_binary_prints(self, tmp_path):
        """The guard's arithmetic is the binary's arithmetic, on a deck that
        runs. Without this the boundary tests above pin a number nothing else
        checks against ``scooter.f90:69``."""
        import re
        from uacpy.models.scooter._plan import _deck_nk
        env = Environment(name='flat', bathymetry=100.0, ssp=1500.0)
        source = Source(depths=25.0, frequencies=100.0)
        receiver = Receiver(depths=np.array([50.0]),
                            ranges=np.array([1000.0]))
        model = Scooter(verbose=False, work_dir=tmp_path, cleanup=False,
                        **self.BAND)
        result = model.compute_tl(env=env, source=source, receiver=receiver)
        predicted = _deck_nk(2000.0, 100.0, 1500.0, 1935.0)
        assert result.run_settings.engine.n_wavenumbers == predicted
        printed = int(re.search(r'Nk =\s+(-?\d+)',
                                (tmp_path / 'model.prt').read_text()).group(1))
        assert printed > 1, (
            f"the binary printed Nk = {printed}: this fixture has to reach a "
            f"count the guard would allow, or it pins nothing")
        assert predicted == printed


class TestScooterKernelTaper:
    """``taper`` is the only knob that reaches the transform rather than the
    deck, and it changes a result by a decibel or two while leaving no other
    trace. Nothing exercised it through ``Scooter`` before these."""

    @staticmethod
    def _grn(nk=2000, c_low=1388.0, c_high=1.0e6):
        return np.linspace(c_low, c_high, nk)

    def test_zero_and_none_both_mean_no_taper(self):
        assert Scooter(taper=0.0).taper == 0.0
        assert Scooter(taper=None).taper == 0.0
        assert taper_bounds(self._grn(),
                            taper=Scooter(taper=0.0).taper) == (None, None)

    def test_the_default_is_off_like_fieldsco(self):
        """``Matlab/Scooter/fieldsco.m:23-32`` sets cmin=1e-10, cmax=1e30 and
        calls tapering "user play (at your own risk)". A default that quietly
        differed would make uacpy disagree with the reference transform."""
        assert Scooter().taper == 0.0
        assert taper_bounds(self._grn(),
                            taper=Scooter().taper) == (None, None)

    @pytest.mark.parametrize('taper', [0.002, 0.01, 0.05, 0.2, 0.49])
    def test_both_edges_are_rolled_off_by_exactly_the_fraction(self, taper):
        """The fraction is taken in k and returned as phase speeds, so both
        edges must move in by the same fraction of the k span."""
        grn = self._grn()
        cmin, cmax = taper_bounds(grn, taper=Scooter(taper=taper).taper)
        c = np.asarray(grn, float)
        lo, hi = 1.0 / c.max(), 1.0 / c.min()          # k/omega at the edges
        span = hi - lo
        assert (1.0 / cmax - lo) / span == pytest.approx(taper, rel=1e-9)
        assert (hi - 1.0 / cmin) / span == pytest.approx(taper, rel=1e-9)
        assert cmin < cmax

    def test_the_bounds_do_not_depend_on_frequency(self):
        """omega cancels, so one broadband sweep uses one pass band."""
        taper = Scooter(taper=0.02).taper
        a = taper_bounds(self._grn(), taper=taper)
        b = taper_bounds(self._grn(nk=8000), taper=taper)
        assert a == pytest.approx(b, rel=1e-6)

    @pytest.mark.parametrize('bad', [-0.01, 0.5, 1.0, float('nan'),
                                     float('inf')])
    def test_out_of_range_is_refused(self, bad):
        with pytest.raises(ConfigurationError, match='taper'):
            Scooter(taper=bad)

    def test_a_dirty_phase_speed_grid_raises_a_uacpy_error(self):
        """``wavenumber_taper`` indexes the RAW grid, so a non-finite entry
        would surface as a numpy ValueError from inside the transform."""
        grn = np.array([1400.0, np.nan, 1500.0, 1600.0, 1700.0])
        with pytest.raises(ConfigurationError, match='non-finite'):
            taper_bounds(grn, taper=Scooter(taper=0.02).taper)

    def test_a_grid_too_small_to_taper_says_so(self):
        """Returning (None, None) silently would make the run indistinguish-
        able from taper=0, with nothing printed even at verbose=True."""
        with pytest.warns(UserWarning, match='untapered'):
            got = taper_bounds(
                np.array([1400.0, 1500.0]),
                taper=Scooter(taper=0.02).taper)
        assert got == (None, None)

    def test_a_single_speed_grid_says_so(self):
        with pytest.warns(UserWarning, match='single speed'):
            got = taper_bounds(
                np.full(64, 1500.0),
                taper=Scooter(taper=0.02).taper)
        assert got == (None, None)

    def test_taper_zero_is_silent_on_a_grid_that_cannot_carry_one(self):
        """Only a REQUESTED taper warns; taper=0 asked for nothing."""
        with recorded_warnings() as caught:
            taper_bounds(np.array([1400.0]),
                         taper=Scooter(taper=0.0).taper)
        assert not caught

    def test_the_taper_reaches_the_result_metadata(self):
        """A 2 dB change that leaves no other trace: two otherwise identical
        results are distinguishable only by this key."""
        import inspect
        assert "taper: float" in inspect.getsource(ScooterSettings)


class TestScooterRefusesAGreenCubeOverTheReaderBudget:
    """``read_grn_file`` allocates the whole ``(nfreq, nsd, nrd, nk)``
    complex64 cube in one ``np.zeros``, and nothing above bounded ``Nk``:
    64 frequencies x 100 receiver depths at the Nk a 25 km receiver grid
    derives is a 12 GiB allocation behind an ordinary-looking broadband
    call. Resolving the run's settings refuses a peak over what the host
    has free, before any file is written; a run over half of it is
    announced, and so is one over 2 GiB on a host whose free memory cannot
    be read."""

    # Bytes per wavenumber sample of the cube at 64 frequencies x 1 source
    # depth x 100 receiver depths; _NK_AT_CAP is a cube of about 2 GiB.
    _PER_NK = 8 * 64 * 1 * 100
    _NK_AT_CAP = UNREADABLE_HOST_BYTES // _PER_NK
    # Peak bytes per wavenumber sample on one range (two cubes and the
    # kernel); floor division puts _NK_AT_PEAK_CAP's peak at most one
    # sample under 2 GiB, so _NK_AT_PEAK_CAP + 1 is over it.
    _PEAK_PER_NK = 2 * _PER_NK + _TRANSFORM_BYTES_PER_ELEMENT
    _NK_AT_PEAK_CAP = UNREADABLE_HOST_BYTES // _PEAK_PER_NK

    @staticmethod
    def _with_free(monkeypatch, nbytes):
        """Pin what the host reports free, so the threshold is not the
        machine's mood. The guard is memory-aware by design, which makes an
        unmocked test pass or fail on whatever else is running."""
        from uacpy.models import _budget
        monkeypatch.setattr(_budget, 'available_memory_bytes',
                            lambda: nbytes)

    def _call(self, nk, nrd=100, nr=1):
        reject_oversized_green_cube(
            nk, 64, 1, nrd, nr,
            rmax_m=5e4, f_deck=2e3, c_low=1406.0, c_high=3436.0)

    def _headroom(self, nk, nrd=100, nr=1):
        return reject_oversized_green_cube(
            nk, 64, 1, nrd, nr,
            rmax_m=5e4, f_deck=2e3, c_low=1406.0, c_high=3436.0)[1]

    def test_a_cube_well_inside_free_memory_is_accepted(self, monkeypatch):
        self._with_free(monkeypatch, 64 * 1024 ** 3)
        self._call(self._NK_AT_CAP)

    def test_a_cube_over_free_memory_is_refused(self, monkeypatch):
        self._with_free(monkeypatch, 1 * 1024 ** 3)
        with pytest.raises(ConfigurationError, match='free'):
            self._call(self._NK_AT_CAP)

    def test_over_half_of_free_memory_warns_but_runs(self, monkeypatch):
        # peak = 2*cube + kernel; make free just over that, under twice it.
        cube = self._PER_NK * self._NK_AT_CAP
        peak = 2 * cube + _TRANSFORM_BYTES_PER_ELEMENT * self._NK_AT_CAP
        self._with_free(monkeypatch, int(peak * 1.5))
        note, message, category = self._headroom(self._NK_AT_CAP)
        assert 'half' in note and 'half' in message
        assert category is NumericsWarning
        # Twice the peak free is not over half of it.
        self._with_free(monkeypatch, 2 * peak)
        assert self._headroom(self._NK_AT_CAP) is None

    def test_the_warning_category_is_one_python_shows_by_default(
            self, monkeypatch):
        """ResourceWarning is on CPython's default ignore list, so a guard
        raising it warns nobody outside pytest (which forces 'always')."""
        import warnings as _w
        env = Environment(name='flat', bathymetry=100.0, ssp=1500.0)
        src = Source(depths=25.0, frequencies=200.0)
        rcv = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
        model = Scooter(verbose=False, c_high=1e9)
        self._with_free(monkeypatch, 64 * 1024 ** 3)
        peak = model.run_settings(env, src, rcv).engine.peak_memory_bytes
        self._with_free(monkeypatch, int(peak * 1.5))
        with _w.catch_warnings(record=True) as caught:
            _w.resetwarnings()            # CPython's defaults, not pytest's
            _w.simplefilter('default')
            model.run_settings(env, src, rcv)
        said = [w for w in caught if 'headroom' in str(w.message)]
        assert said, 'the guard warned nobody under default filters'
        assert not issubclass(said[0].category, ResourceWarning)

    def test_the_kernel_is_counted_not_only_the_cube(self, monkeypatch):
        """The estimate covers the transform kernel, not just the cube: a
        deck whose cube fits free memory but whose (nk x nr) kernel does not
        is refused, and the message says which term it was."""
        self._with_free(monkeypatch, 3 * 1024 ** 3)
        with pytest.raises(ConfigurationError, match='transform kernel'):
            self._call(self._NK_AT_CAP, nrd=1, nr=200_000)

    def test_an_unreadable_host_announces_a_peak_over_2_gib(
            self, monkeypatch):
        """With no MemAvailable to size against, a peak at 2 GiB is silent
        and one sample past it is announced, never refused: nothing
        measured says it cannot fit."""
        self._with_free(monkeypatch, None)
        assert self._headroom(self._NK_AT_PEAK_CAP, nr=1) is None
        note, message, _category = self._headroom(
            self._NK_AT_PEAK_CAP + 1, nr=1)
        assert 'cannot be read' in message and 'Nk' in message

    def test_an_over_budget_broadband_deck_is_refused_before_writing(
            self, tmp_path, monkeypatch):
        # Pinned free memory: otherwise this passes or fails on how much RAM
        # the machine happens to have, which is not what it is testing.
        self._with_free(monkeypatch, 4 * 1024 ** 3)
        model = Scooter(verbose=False, c_low=1406.0, c_high=3436.0,
                        work_dir=tmp_path, cleanup=False)
        with pytest.raises(ConfigurationError, match='Nk'):
            model.run(
                Environment(name='deep', bathymetry=5000.0, ssp=1480.0),
                Source(depths=100.0, frequencies=500.0),
                Receiver(depths=np.linspace(10.0, 4900.0, 100),
                         ranges=np.array([25000.0])),
                RunMode.BROADBAND,
                frequencies=np.linspace(500.0, 2000.0, 64))
        assert not (tmp_path / 'model.env').exists()

    def test_a_small_narrowband_deck_is_resolved(self, monkeypatch):
        self._with_free(monkeypatch, 4 * 1024 ** 3)
        settings = Scooter(verbose=False).run_settings(
            Environment(name='flat', bathymetry=100.0, ssp=1500.0),
            Source(depths=25.0, frequencies=200.0),
            Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])))
        assert settings.engine.n_wavenumbers >= 2


class TestSteepPathsBeyondTheDefaultCutAreAnnounced:
    """The auto-derived c_high keeps paths only to arccos(c/c_high) grazing;
    a receiver seeing a steeper direct or surface path lost it silently —
    23 dB at 0.2 km in a Lloyd geometry over a transparent bottom."""

    @staticmethod
    def _env():
        return make_pekeris(bathymetry=400.0, sound_speed=1500.0,
                            density=1.027, attenuation=0.0)

    def test_a_near_receiver_warns_and_the_remedy_is_exact(self):
        src = Source(depths=25.0, frequencies=150.0)
        rcv = Receiver(depths=[200.0], ranges=[200.0, 500.0, 1200.0])
        with pytest.warns(UserWarning, match="c_high=1e9"):
            Scooter().compute_tl(self._env(), src, rcv)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fixed = Scooter(c_high=1e9, taper=0.05).compute_tl(
                self._env(), src, rcv)
        # Exact image solution (JKPS eq. 1.19): 43.03 / 55.75 / 61.18 dB.
        np.testing.assert_allclose(np.asarray(fixed.dB).ravel(),
                                   [43.03, 55.75, 61.18], atol=0.1)

    def test_a_pinned_c_high_is_trusted(self):
        src = Source(depths=25.0, frequencies=150.0)
        rcv = Receiver(depths=[200.0], ranges=[200.0])
        with recorded_warnings() as rec:
            Scooter(c_high=1575.0).compute_tl(self._env(), src, rcv)
        assert not any("c_high=1e9" in str(w.message) for w in rec)


class _Launched(Exception):
    """Raised by the launch spy: the call reached the binary."""


def _pekeris():
    return make_pekeris(name='pekeris', sound_speed=1800.0, attenuation=0.3)


def _near_source():
    return Source(depths=25.0, frequencies=100.0)


def _grid():
    return Receiver(depths=np.array([20.0, 60.0]),
                    ranges=np.array([1000.0, 3000.0]))


class TestTheThreeEntryPointsRefuseAlike:
    """RA-CONTRACT-18 / ARCH-9 for Scooter's own refusals: they live in its
    ``_validate_engine`` (carriers and knobs) and ``_resolve_engine_settings``
    (the deck), which ``validate_inputs``, ``run_settings`` and ``run`` all
    run, so the three raise the same exception with the same message and
    ``run`` launches nothing."""

    @pytest.mark.parametrize('label', [
        "interp_ssp='quad'",
        'a taper reassigned out of range',
        'c_low pinned above the derived c_high',
        'fewer than two wavenumber samples',
        'n_mesh under the mesh floor',
        'TIME_SERIES without a source pulse',
    ])
    def test_validate_inputs_run_settings_and_run_raise_the_same(
            self, label, monkeypatch):
        env, src, rcv = _pekeris(), _near_source(), _grid()
        model_kw, mode = {}, None
        if label == "interp_ssp='quad'":
            model_kw = dict(interp_ssp='quad')
        elif label == 'c_low pinned above the derived c_high':
            model_kw = dict(c_low=2000.0)
        elif label == 'fewer than two wavenumber samples':
            model_kw = dict(c_low=1500.0, c_high=1935.0)
            rcv = Receiver(depths=np.array([50.0]), ranges=np.array([8.3]))
        elif label == 'n_mesh under the mesh floor':
            model_kw = dict(n_mesh=5)
        elif label == 'TIME_SERIES without a source pulse':
            mode = RunMode.TIME_SERIES
        model = Scooter(verbose=False, **model_kw)
        if label == 'a taper reassigned out of range':
            model.taper = 0.7

        def _spy(*a, **k):
            raise _Launched
        monkeypatch.setattr(model, '_run_subprocess', _spy)
        outcomes = []
        for call in ('validate_inputs', 'run_settings', 'run'):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    getattr(model, call)(env, src, rcv, mode)
            except _Launched:
                outcomes.append('launched')
            except Exception as exc:          # noqa: BLE001
                outcomes.append((type(exc).__name__, str(exc)))
            else:
                outcomes.append('accepted')
        assert outcomes[0] not in ('accepted', 'launched'), outcomes
        assert outcomes[0] == outcomes[1] == outcomes[2], outcomes


class TestTheSettingsRecordTheDeck:
    """``Scooter().run_settings(...).engine`` is the :class:`ScooterSettings`
    the deck is written from: the phase-speed window and where each bound
    came from (the rule RA-WAVE-8 states), RMax, the mesh, Nk."""

    def test_the_record_round_trips_and_pickles(self):
        import pickle
        from uacpy.models.scooter import ScooterSettings
        engine = Scooter(verbose=False).run_settings(
            _pekeris(), _near_source(), _grid()).engine
        assert isinstance(engine, ScooterSettings)
        for copy in (ScooterSettings.from_dict(engine.to_dict()),
                     pickle.loads(pickle.dumps(engine))):
            assert copy.to_dict() == engine.to_dict()

    @pytest.mark.parametrize('seabed', ['vacuum', 'rigid', 'precalc'])
    def test_a_seabed_without_a_half_space_speed_leaves_c_high_unbounded(
            self, seabed, tmp_path):
        from uacpy.core import BoundaryProperties
        from uacpy.models._window import DEFAULT_C_MAX_UNBOUNDED
        kw = {}
        if seabed == 'precalc':
            table = tmp_path / 'seabed.irc'
            table.write_text(
                "'seabed' 50.0\n2\n"
                "  0.1000000E+00  0.2000000E+00  0.0000000E+00  0.3000000E+00"
                "  0.0000000E+00    0\n"
                "  0.2000000E+00  0.2500000E+00  0.0000000E+00  0.3500000E+00"
                "  0.0000000E+00    0\n")
            kw = dict(reflection_file=str(table))
        env = Environment(name=seabed, bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(acoustic_type=seabed,
                                                    **kw))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            engine = Scooter(verbose=False).run_settings(
                env, _near_source(), _grid()).engine
        assert engine.c_high == DEFAULT_C_MAX_UNBOUNDED
        assert "'precalc'" in engine.c_high_origin

    def test_a_per_depth_loop_sizes_the_memory_for_one_depth(self):
        """BROADBAND launches one source depth at a time, so the cube a
        launch reads holds one; COHERENT_TL writes every depth into one
        deck."""
        model = Scooter(verbose=False)
        freqs = np.linspace(80.0, 120.0, 5)
        one, two = (model.run_settings(
            _pekeris(), Source(depths=depths, frequencies=100.0), _grid(),
            RunMode.BROADBAND, frequencies=freqs).engine.peak_memory_bytes
            for depths in ([25.0], [25.0, 50.0]))
        assert one == two
        narrow_one, narrow_two = (model.run_settings(
            _pekeris(), Source(depths=depths, frequencies=100.0),
            _grid()).engine.peak_memory_bytes
            for depths in ([25.0], [25.0, 50.0]))
        assert narrow_two > narrow_one

    def test_a_half_space_caps_c_high_at_its_speed(self):
        engine = Scooter(verbose=False).run_settings(
            _pekeris(), _near_source(), _grid()).engine
        assert engine.c_high == pytest.approx(1.05 * 1800.0)
        assert engine.c_high_origin.startswith('1.05 × max(')

    def test_the_docstrings_state_the_window_rules(self):
        """RA-WAVE-8 / RA-WAVE-16: the unbounded c_high names every
        seabed it applies to, and the default c_low says it leaves the
        interface waves out."""
        def entry(doc, start):
            return ' '.join(doc[doc.index(start):doc.index('n_mesh :')]
                            .split())
        # Each engine documents its constructor once (class or __init__).
        found = [(doc, start)
                 for doc, start in ((Scooter.__doc__, 'c_low, c_high :'),
                                    (Scooter.__init__.__doc__, 'c_low :'))
                 if doc and start in doc and 'n_mesh :' in doc]
        assert found, 'no docstring documents c_low/c_high'
        for doc, start in found:
            text = entry(doc, start)
            for seabed in ('vacuum', 'rigid', "'file'", "'precalc'"):
                assert seabed in text, (seabed, text)
            assert 'Scholte' in text and 'shear speed' in text, text


@pytest.mark.requires_binary
class TestTheDeckIsWrittenFromTheResolvedSettings:
    """The deck and the binary's own echo (``.prt``) carry what
    ``run_settings`` shows: the cLow/cHigh line, RMax (km) on the line after
    it (``write_phase_speed_and_rmax``), ``Nk``, and the frequency vector of
    a broadband run."""

    @pytest.mark.parametrize('mode, kw', [
        (RunMode.COHERENT_TL, {}),
        (RunMode.BROADBAND, dict(frequencies=np.linspace(80.0, 120.0, 5))),
    ])
    def test_deck_and_echo_equal_the_settings(self, mode, kw, tmp_path):
        import re
        model = Scooter(verbose=False, work_dir=tmp_path, cleanup=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = model.run(_pekeris(), _near_source(), _grid(), mode, **kw)
        engine = result.run_settings.engine
        lines = (tmp_path / 'model.env').read_text().splitlines()
        speeds = f'{engine.c_low:.1f} {engine.c_high:.1f}'
        assert speeds in lines
        assert float(lines[lines.index(speeds) + 1]) == pytest.approx(
            engine.rmax_m / 1000.0)
        prt = (tmp_path / 'model.prt').read_text()
        assert int(re.search(r'Nk =\s+(-?\d+)', prt).group(1)) == engine.n_wavenumbers
        if mode == RunMode.BROADBAND:
            assert result.run_settings.frequencies.size == 5
            assert engine.deck_max_frequency == 120.0


class TestNoticesComeFromRunNotFromValidateInputs:
    """The steep-path notice states how a run will go with the resolved
    window and refuses nothing, so ``run_settings`` (and ``run``) give it
    and ``validate_inputs`` does not."""

    @staticmethod
    def _call(entry):
        env = TestSteepPathsBeyondTheDefaultCutAreAnnounced._env()
        with recorded_warnings() as caught:
            out = getattr(Scooter(verbose=False), entry)(
                env, Source(depths=25.0, frequencies=150.0),
                Receiver(depths=[200.0], ranges=[200.0, 500.0]))
        return out, [str(w.message) for w in caught if 'c_high=1e9' in
                     str(w.message)]

    def test_run_settings_announces_the_steep_path_once(self):
        assert len(self._call('run_settings')[1]) == 1

    @pytest.mark.requires_binary
    def test_run_announces_the_steep_path_once(self):
        assert len(self._call('run')[1]) == 1

    def test_validate_inputs_is_silent(self):
        assert self._call('validate_inputs')[1] == []

    @pytest.mark.requires_binary
    def test_a_per_depth_run_announces_it_once_for_every_depth(self):
        """BROADBAND launches once per source depth; the notice is a fact
        of the call's settings, judged over every depth, and given once
        before the first launch."""
        env = TestSteepPathsBeyondTheDefaultCutAreAnnounced._env()
        with recorded_warnings() as caught:
            Scooter(verbose=False).run(
                env, Source(depths=[25.0, 30.0], frequencies=150.0),
                Receiver(depths=[200.0], ranges=[200.0, 500.0]),
                RunMode.BROADBAND, frequencies=[140.0, 150.0, 160.0])
        said = [str(w.message) for w in caught
                if 'c_high=1e9' in str(w.message)]
        assert len(said) == 1, said

    def test_the_settings_record_the_condition(self):
        settings, said = self._call('run_settings')
        notes = [n.note for n in settings.engine.notices if n.note]
        assert len(notes) == 1 and 'drops paths steeper than' in notes[0]
        assert [n.message for n in settings.engine.notices] == said
        assert 'drops paths steeper than' in repr(settings)

    def test_a_pinned_c_high_records_nothing(self):
        env = TestSteepPathsBeyondTheDefaultCutAreAnnounced._env()
        settings = Scooter(verbose=False, c_high=1575.0).run_settings(
            env, Source(depths=25.0, frequencies=150.0),
            Receiver(depths=[200.0], ranges=[200.0]))
        assert settings.engine.notices == ()



@pytest.mark.parametrize('c_low, refused', [(1500.0, True), (1499.9, False)])
def test_a_pinned_pair_is_refused_at_equality(c_low, refused):
    """Two pinned bounds must satisfy c_low < c_high strictly (Scooter and
    SPARC share ``models/_window.check_pinned_window``)."""
    from uacpy.models import SPARC
    for cls in (Scooter, SPARC):
        if refused:
            with pytest.raises(ConfigurationError, match='c_low < c_high'):
                cls(c_low=c_low, c_high=1500.0)
        else:
            assert cls(c_low=c_low, c_high=1500.0).c_low == c_low


class TestScooterRefusesAPrecalcSurface:
    """No Acoustics Toolbox binary reads a top ``.irc`` table
    (misc/RefCoef.f90:92 reads it for the bottom only), and scooter.exe
    interpolates the empty table and dies with SIGSEGV
    (Scooter/scooter.f90:357-358). The surface is refused before launch; a
    vacuum surface over the same seabed validates."""

    @staticmethod
    def _env(surface_type, tmp_path):
        from uacpy.core import BoundaryProperties, Surface
        irc = tmp_path / 'top.irc'
        irc.write_text('1\n0.0 1.0 0.0\n')
        top = (BoundaryProperties(acoustic_type='precalc',
                                  reflection_file=str(irc))
               if surface_type == 'precalc'
               else BoundaryProperties(acoustic_type='vacuum'))
        return Environment(
            bathymetry=100.0, ssp=1500.0, surface=Surface(nodes=[top]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5))

    def test_a_precalc_surface_is_refused_before_launch(self, tmp_path):
        from uacpy.core.exceptions import UnsupportedFeatureError
        with pytest.raises(UnsupportedFeatureError,
                           match=r"'precalc' \(\.irc\) sea surface"):
            Scooter(verbose=False).validate_inputs(
                self._env('precalc', tmp_path), _near_source(), _grid())

    def test_a_vacuum_surface_validates(self, tmp_path):
        Scooter(verbose=False).validate_inputs(
            self._env('vacuum', tmp_path), _near_source(), _grid())
