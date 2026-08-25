"""Scooter wavenumber-integration-focused tests."""

import pytest
import numpy as np

from uacpy.core.results import Field
from uacpy.models import Scooter
from uacpy.models.base import RunMode
from uacpy.core import Environment, Source, Receiver
from uacpy.core.exceptions import ConfigurationError

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
        # sample different wavenumber grids (rmax_multiplier 3.0 vs 2.0), so
        # the comparison bounds the median bias rather than pinning equality.
        nb = Scooter(verbose=False).compute_tl(env=env, source=source,
                                               receiver=receiver)
        got = np.asarray(result.at(frequency=100.0).to_db().data, dtype=float)
        ref = np.asarray(nb.db, dtype=float)
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
    ``Scooter._assemble_field_from_grn`` reads ``source.source_type``. A
    duplicate on the model would let the deck and the carrier disagree about
    what was radiated.
    """
    with pytest.raises(TypeError):
        Scooter(source_type='R')


def test_scooter_constructor_rejects_field_interp():
    """``field_interp`` named an FLP option for ``fields.exe``, which uacpy
    never runs — the k→r transform is done in-tree."""
    with pytest.raises(TypeError):
        Scooter(field_interp='P')


def test_grn_transform_method_is_the_direct_dft():
    """The transform is a trapezoidal-rule DFT (``fieldsco.m:5``), not an FFT,
    and ``'fft_hankel'`` is not an accepted name for it."""
    from uacpy.io.grn_reader import grn_to_field

    with pytest.raises(ConfigurationError, match="direct_dft"):
        grn_to_field({}, np.array([1000.0]), method='fft_hankel')


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
                Scooter(verbose=False).run(env, source, self._receiver()).db)
        with pytest.warns(UserWarning, match="r = 0"):
            kraken_tl = np.asarray(
                Kraken(verbose=False).compute_tl(
                    env, source, self._receiver()).db)

        assert np.all(np.isnan(scooter_tl[:, 0]))
        assert np.all(np.isnan(kraken_tl[:, 0]))
        assert np.all(np.isfinite(scooter_tl[:, 1:]))
        assert np.all(np.isfinite(kraken_tl[:, 1:]))


def test_a_receiver_with_no_positive_range_is_refused_before_launch():
    """Scooter's spectral RMax derives from the maximum receiver range
    (``RMax = range_max × rmax_multiplier``), so a receiver whose ranges
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
        from uacpy.core.bottom import Bottom, SeabedColumn, SedimentLayer
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
    def _grn(atten, k, title='SCOOTER - test'):
        return {'is_sparc': False, 'title': title, 'atten': atten,
                'freq': 100.0, 'cVec': np.array([1500.0])}, k

    def test_a_zero_header_is_honoured(self):
        """The solver wrote 0 because TopOpt(7:7) = '0'; Δk must not come back."""
        from uacpy.io.grn_reader import _stab_attenuation
        grn, k = self._grn(0.0, np.array([1.0, 1.0 + 1.5745e-4, 1.0 + 3.149e-4]))
        assert _stab_attenuation(grn, k) == 0.0

    def test_a_non_zero_header_is_re_derived_per_frequency(self):
        """With the stabiliser on, the header carries only the first frequency's
        Δk, so the k vector is the authority."""
        from uacpy.io.grn_reader import _stab_attenuation
        dk = 1.5745e-4
        grn, k = self._grn(9.9e-9, np.array([1.0, 1.0 + dk, 1.0 + 2 * dk]))
        assert _stab_attenuation(grn, k) == pytest.approx(dk)

    def test_sparc_is_unaffected(self):
        from uacpy.io.grn_reader import _stab_attenuation
        grn, k = self._grn(1.0, np.array([1.0, 2.0]))
        grn['is_sparc'] = True
        assert _stab_attenuation(grn, k) == 0.0

    def test_turning_the_stabiliser_off_warns_with_its_cost(self):
        """Removing the contour offset puts the modal poles back on the
        integration path, which a correct transform cannot repair."""
        import warnings as _w
        with pytest.warns(UserWarning, match='modal poles'):
            Scooter(stabilizing_attenuation_off=True)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter('always')
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
    ``RMax = receiver.ranges.max() × rmax_multiplier`` with the multiplier
    defaulting to 2.0 narrowband / 3.0 broadband, and
    ``c_low = 0.95 × min(SSP)`` when only ``c_high`` is pinned. The deck is
    written by ``_write_scooter_env`` without launching the binary; the
    cLow/cHigh line and the RMax (km) line are consecutive
    (``write_phase_speed_and_rmax``)."""

    @staticmethod
    def _env():
        from uacpy.core import BoundaryProperties
        return Environment(
            name='deck', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5))

    @staticmethod
    def _deck_lines(tmp_path, model, run_mode=RunMode.COHERENT_TL,
                    frequencies=None):
        deck = tmp_path / 'deck.env'
        model._write_scooter_env(
            deck, TestScooterDeckResolution._env(),
            Source(depths=50.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]),
                     ranges=np.array([1000.0, 3000.0])),
            frequencies=frequencies, run_mode=run_mode)
        return deck.read_text().splitlines()

    @staticmethod
    def _rmax_km_after_speed_line(lines, speed_line):
        assert speed_line in lines, (
            f"expected phase-speed line {speed_line!r} in deck: {lines}")
        return float(lines[lines.index(speed_line) + 1])

    def test_narrowband_rmax_is_twice_the_receiver_max(self, tmp_path):
        # 0.95×1500 = 1425.0, 1.05×1800 = 1890.0; RMax = 3000 m × 2.0.
        lines = self._deck_lines(tmp_path, Scooter(verbose=False))
        assert self._rmax_km_after_speed_line(
            lines, '1425.0 1890.0') == pytest.approx(6.0)

    def test_broadband_rmax_is_three_times_the_receiver_max(self, tmp_path):
        lines = self._deck_lines(
            tmp_path, Scooter(verbose=False), run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(80.0, 120.0, 5))
        assert self._rmax_km_after_speed_line(
            lines, '1425.0 1890.0') == pytest.approx(9.0)

    def test_pinned_rmax_multiplier_wins_in_both_modes(self, tmp_path):
        model = Scooter(rmax_multiplier=5.0, verbose=False)
        narrow = self._deck_lines(tmp_path, model)
        broad = self._deck_lines(tmp_path, model, run_mode=RunMode.BROADBAND,
                                 frequencies=np.linspace(80.0, 120.0, 5))
        assert self._rmax_km_after_speed_line(
            narrow, '1425.0 1890.0') == pytest.approx(15.0)
        assert self._rmax_km_after_speed_line(
            broad, '1425.0 1890.0') == pytest.approx(15.0)

    def test_c_low_auto_derives_with_c_high_pinned(self, tmp_path):
        # Only c_high pinned: c_low still resolves to 0.95 × min(SSP).
        lines = self._deck_lines(
            tmp_path, Scooter(c_high=1700.0, verbose=False))
        assert '1425.0 1700.0' in lines

    def test_documented_factors_are_the_code_factors(self):
        # scooter.md and DOCUMENTATION.md state 0.95 / 1.05 as literals; this
        # pins the constants so doc-vs-code drift fails a test.
        from uacpy.core.constants import C_LOW_FACTOR, C_HIGH_FACTOR
        assert C_LOW_FACTOR == 0.95
        assert C_HIGH_FACTOR == 1.05


class TestScooterSpectrumOption:
    """``spectrum`` names the wavenumber branch the k→r transform integrates
    (``docs/models/scooter.md`` constructor table); the constructor maps it to
    the one-letter code handed to ``grn_to_field`` / the ``.flp`` convention."""

    @pytest.mark.parametrize('name,code', [
        ('positive', 'P'), ('negative', 'N'), ('both', 'B')])
    def test_spectrum_name_maps_to_code(self, name, code):
        assert Scooter(spectrum=name, verbose=False)._spectrum_code == code

    def test_unknown_spectrum_raises(self):
        with pytest.raises(ConfigurationError, match='spectrum'):
            Scooter(spectrum='full', verbose=False)


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
        return np.asarray(result.db, dtype=float)

    @pytest.mark.requires_binary
    def test_sub_floor_n_mesh_is_bit_identical_to_the_floor(self):
        assert np.array_equal(self._tl(40), self._tl(100))

    @pytest.mark.requires_binary
    def test_above_floor_n_mesh_moves_the_answer(self):
        assert not np.array_equal(self._tl(150), self._tl(100))


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
        with pytest.raises(ConfigurationError) as err:
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
                          bottom=_halfspace(3000.0, density=2.0,
                                            attenuation=0.1))
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([50.0]), ranges=np.array([2000.0]))
        result = Scooter(verbose=False).run(
            env, src, rcv, run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(80.0, 120.0, 5))

        assert result.metadata['c_max'] == pytest.approx(3000.0)

        trace = result.to_time_trace(depth=50.0, range=2000.0)
        t = np.asarray(trace.coords['time'], dtype=float)
        # T_window = 1/df = 0.1 s, so the window opens half a window ahead
        # of the r / c_max = 0.667 s fastest possible arrival — not at the
        # 1.283 s a 1500 m/s default anchor gives.
        assert t[0] == pytest.approx(2000.0 / 3000.0 - 0.05, abs=0.02)


@pytest.mark.requires_binary
class TestScooterRefusesAPhaseSpeedBandInvertedByOnePinnedBound:
    """The constructor can only compare two pinned bounds. One pinned bound is
    comparable once the other is derived from the env, which happens in
    ``_write_scooter_env``; on the 100 m / 1500 m/s guide the auto band is
    (1425, 1680) m/s, so ``c_low=2000`` alone writes CLOW > CHIGH.
    ``ReadEnvironmentMod.f90:135`` then stops the binary after the deck has
    been written and the process spawned.
    """

    @staticmethod
    def _write(tmp_path, **kwargs):
        env = Environment(name='flat', bathymetry=100.0, ssp=1500.0)
        Scooter(verbose=False, **kwargs)._write_scooter_env(
            tmp_path / 'case.env', env,
            Source(depths=25.0, frequencies=200.0),
            Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])))

    def test_a_pinned_c_low_above_the_derived_c_high_names_c_low(self, tmp_path):
        with pytest.raises(ConfigurationError, match='pinned c_low=2000'):
            self._write(tmp_path, c_low=2000.0)
        assert not (tmp_path / 'case.env').exists()

    def test_a_pinned_c_high_below_the_derived_c_low_names_c_high(self, tmp_path):
        with pytest.raises(ConfigurationError, match='pinned c_high=100'):
            self._write(tmp_path, c_high=100.0)

    def test_a_band_that_brackets_the_derived_bounds_writes_the_deck(self, tmp_path):
        self._write(tmp_path, c_low=1200.0)
        assert (tmp_path / 'case.env').exists()


class TestScooterRefusesAWavenumberGridItCannotSpace:
    """``scooter.f90:69`` derives ``Nk = INT(2000*RMax_km*(kMax-kMin)/pi)`` and
    has no test of the value — the only ``IF`` naming ``Nk`` is the allocation
    status at ``:74``. ``Nk = 1`` then divides by ``Nk - 1 = 0`` at ``:77`` and
    ``:125``, so the binary writes an all-NaN Green's function and exits 0;
    ``Nk = 0`` writes one with no samples at all. Both refused before launch,
    as ``bounce.py`` and ``sparc.py`` refuse the same arithmetic.

    On a 100 m isovelocity guide at 100 Hz with the band pinned to
    (1500, 1935) m/s, ``RMax = ranges.max() x 2`` puts the ``Nk = 1`` / ``2``
    boundary between a 16.6 m and a 16.7 m furthest receiver.
    """

    BAND = dict(c_low=1500.0, c_high=1935.0)

    @staticmethod
    def _write(tmp_path, r_max, **kwargs):
        env = Environment(name='flat', bathymetry=100.0, ssp=1500.0)
        Scooter(verbose=False, **kwargs)._write_scooter_env(
            tmp_path / 'case.env', env,
            Source(depths=25.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]), ranges=np.array([r_max])))

    def test_a_single_sample_grid_is_refused_before_the_deck_is_written(
            self, tmp_path):
        with pytest.raises(ConfigurationError, match=r'Nk = 1 wavenumber'):
            self._write(tmp_path, 16.6, **self.BAND)
        assert not (tmp_path / 'case.env').exists()

    def test_two_samples_write_the_deck(self, tmp_path):
        """The high side of the same boundary — 0.1 m further out."""
        self._write(tmp_path, 16.7, **self.BAND)
        assert (tmp_path / 'case.env').exists()

    def test_an_empty_grid_is_refused_before_the_deck_is_written(
            self, tmp_path):
        with pytest.raises(ConfigurationError, match=r'Nk = 0 wavenumber'):
            self._write(tmp_path, 8.3, **self.BAND)
        assert not (tmp_path / 'case.env').exists()

    def test_the_two_counts_are_refused_for_their_own_reasons(self, tmp_path):
        """``Nk = 1`` and ``Nk = 0`` fail differently in the Fortran, so the
        message must not describe one as the other."""
        with pytest.raises(ConfigurationError) as one:
            self._write(tmp_path, 16.6, **self.BAND)
        with pytest.raises(ConfigurationError) as zero:
            self._write(tmp_path, 8.3, **self.BAND)
        assert 'divides by zero' in str(one.value)
        assert 'divides by zero' not in str(zero.value)

    def test_the_remediation_names_the_knobs_that_raise_nk(self, tmp_path):
        """``Nk`` grows with ``RMax``, and ``RMax = ranges.max() x
        rmax_multiplier`` — so advice to shorten the receiver ranges lowers
        the count that is already too low."""
        with pytest.raises(ConfigurationError) as exc:
            self._write(tmp_path, 16.6, **self.BAND)
        text = str(exc.value)
        assert 'rmax_multiplier' in text
        assert 'c_low/c_high' in text
        assert 'shorten the receiver ranges' not in text

    def test_the_predicted_count_is_the_count_the_binary_prints(self, tmp_path):
        """The guard's arithmetic is the binary's arithmetic, on a deck that
        runs. Without this the boundary tests above pin a number nothing else
        checks against ``scooter.f90:69``."""
        import re
        from uacpy.models.scooter import _deck_nk
        env = Environment(name='flat', bathymetry=100.0, ssp=1500.0)
        source = Source(depths=25.0, frequencies=100.0)
        receiver = Receiver(depths=np.array([50.0]),
                            ranges=np.array([1000.0]))
        model = Scooter(verbose=False, work_dir=tmp_path, cleanup=False,
                        **self.BAND)
        model.compute_tl(env=env, source=source, receiver=receiver)
        predicted = _deck_nk(2000.0, 100.0, 1500.0, 1935.0)
        printed = int(re.search(r'Nk =\s+(-?\d+)',
                                (tmp_path / 'model.prt').read_text()).group(1))
        assert printed > 1, (
            f"the binary printed Nk = {printed}: this fixture has to reach a "
            f"count the guard would allow, or it pins nothing")
        assert predicted == printed


from uacpy.core.bottom import BoundaryProperties


def _halfspace(sound_speed, **kwargs):
    return BoundaryProperties(
        acoustic_type='half-space', sound_speed=sound_speed,
        density=kwargs.pop('density', 1.8),
        attenuation=kwargs.pop('attenuation', 0.3), **kwargs)
