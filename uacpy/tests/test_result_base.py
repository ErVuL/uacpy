"""What every result type shares (``uacpy.core.results._base``).

Arrays copied on ingest, the validated phase reference and run mode, the
quantity registry, the dict round trip of every result type, the settings a
result ran with, and the array products (``Covariance``, ``Replicas``).
"""

import inspect
import numpy as np
import pytest
import uacpy
from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field
from uacpy.core.results import PhaseReference
from uacpy.core.results import ReflectionCoefficient
from uacpy.core.results._base import Result
from uacpy.core.results.stack import ResultStack
from uacpy.core.source import Source
from uacpy.tests._synthetic_fields import _field
from uacpy.tests._synthetic_fields import _two_path_grid


class TestResultIngestCopiesArrays:
    """Every ``Result`` copies on ingest, and ``Field.to_dict`` is a real
    snapshot — a caller mutating their source array can never reach inside a
    result, and a cached dict never aliases the field it came from."""

    @staticmethod
    def _field():
        from uacpy.core.results import Field
        return Field(
            data=np.ones((2, 3), dtype=complex),
            coords={'depth': np.array([10.0, 20.0]),
                    'range': np.array([100.0, 200.0, 300.0])},
            model='Test', source_depths=np.array([50.0]),
            frequencies=np.array([100.0]))

    def test_result_copies_identity_arrays_on_ingest(self):
        from uacpy.core.results import Field
        sd = np.array([50.0])
        fr = np.array([100.0])
        f = Field(data=np.ones((1, 1)), coords={'depth': [0.0], 'range': [0.0]},
                  source_depths=sd, frequencies=fr)
        sd[0] = 999.0
        fr[0] = 999.0
        assert f.source_depths[0] == 50.0
        assert f.frequencies[0] == 100.0

    def test_to_dict_is_a_snapshot(self):
        f = self._field()
        d = f.to_dict()
        for key, live in (('data', f.data), ('frequencies', f.frequencies),
                          ('source_depths', f.source_depths)):
            assert not np.shares_memory(d[key], live), key
        for name, vec in f.coords.items():
            assert not np.shares_memory(d['coords'][name], vec), name

    def test_covariance_and_replicas_copy_on_ingest(self):
        from uacpy.core.results import Covariance, Replicas
        cov_src = np.zeros((1, 2, 2), dtype=complex)
        cov = Covariance(covariance=cov_src, model='OASN')
        cov_src[0, 0, 0] = 99.0
        assert cov.covariance[0, 0, 0] == 0.0

        rep_src = np.zeros((1, 1, 1, 1, 2), dtype=complex)
        rep = Replicas(replicas=rep_src,
                       candidates={'depth': [0.0], 'x': [0.0], 'y': [0.0]},
                       model='OASN')
        rep_src[0, 0, 0, 0, 0] = 7.0
        assert rep.replicas[0, 0, 0, 0, 0] == 0.0


class TestPhaseReferenceIsValidatedAtIngest:
    """A value outside :class:`PhaseReference` compares unequal to
    ``'time_domain_native'`` and so walks straight through the IFFT guard
    that reads it. The value is checked for membership but stored exactly as
    passed, so a plain string stays a plain string."""

    def test_an_unknown_phase_reference_raises_a_typed_error(self):
        with pytest.raises(ConfigurationError,
                           match="not a known phase convention"):
            Result(model='X', phase_reference='travelling_wve')

    def test_the_error_names_the_accepted_values(self):
        with pytest.raises(ConfigurationError,
                           match='is not a known phase convention') as exc:
            Field(data=np.zeros((2, 3)),
                  coords={'depth': np.array([10.0, 20.0]),
                          'range': np.array([1.0, 2.0, 3.0])},
                  phase_reference='travelling')
        assert 'travelling_wave' in str(exc.value)
        assert 'time_domain_native' in str(exc.value)

    def test_a_plain_string_is_stored_without_coercion(self):
        result = Result(model='X', phase_reference='travelling_wave')
        assert type(result.phase_reference) is str
        assert str(result.phase_reference).endswith('travelling_wave')

    def test_an_enum_member_is_stored_without_conversion(self):
        result = Result(model='X',
                        phase_reference=PhaseReference.TRAVELLING_WAVE)
        assert result.phase_reference is PhaseReference.TRAVELLING_WAVE

    def test_none_is_accepted(self):
        assert Result(model='X').phase_reference is None

    def test_a_stored_value_survives_the_field_dict_round_trip(self):
        field = Field(data=np.zeros((2, 3)),
                      coords={'depth': np.array([10.0, 20.0]),
                              'range': np.array([1.0, 2.0, 3.0])},
                      phase_reference='time_domain_native')
        assert (Field.from_dict(field.to_dict()).phase_reference
                == 'time_domain_native')


class TestTheOassReverberationCitationNamesTheRoutineOnOptionRsPath:
    """The quantity registry's docstring (:mod:`uacpy.core.results.quantities`)
    explains the reverberation quantity by citing the OASES lines that convert
    it to dB. Getting that
    address right takes the DATA PATH, not the block's contents.

    ``oassun26.f`` carries two byte-identical "CONVERT TO dB" blocks —
    ``REVRAN`` at :633-638 and ``REVINT`` at :853-858 — so every content
    assertion passes on either, and a citation naming the wrong routine reads
    as correct. The entry has been wrong twice: first ``:876-880``, which is a
    comment, a PARAMETER and three INCLUDEs and converts nothing, then
    ``:633-638``, which converts but is the wrong routine.

    ``REVINT`` is the one. It writes ``CFFs``, which ``unoass21.f:38``
    equivalences to the ``XS`` that ``PLTLOS`` plots into the ``.plt`` uacpy
    reads, and option ``'r'`` sets ``PLTL`` (``unoass21.f:607-609``) which is
    the branch that calls it. ``REVRAN`` writes ``CFF(1,1)`` (equivalenced to
    ``X``), accumulates a cross-range covariance rather than an intensity, and
    is reached only from the ``CCONTU`` contour branch that the **capital**
    option ``'C'`` enables (``unoass21.f:626-628``). The letters are
    case-sensitive and the lowercase one is a third output again: ``'c'`` sets
    ``ICONTU``, the depth-integrand contours (``unoass21.f:602-604``), which
    reach neither routine. The Fortran blocks themselves (their contents, the
    array each writes, the routine each sits in) are pinned once, on the model
    side, in
    ``test_oass.py::TestTheReverberationCitationNamesTheRoutineOnOptionRsPath``;
    this class pins the registry's sentence and the option letters.
    """

    @staticmethod
    def _src(name):
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent / 'third_party' /
                'oases' / 'src' / name)
        return path.read_text().splitlines()

    @staticmethod
    def _sentence():
        from uacpy.core.results import quantities
        return quantities.__doc__

    @pytest.mark.requires_oases
    def test_the_two_accumulators_are_different_quantities(self):
        # Why the -5 is a -10 here and why the other block is not this one:
        # REVINT multiplies one sample by its own conjugate (an intensity),
        # REVRAN crosses two different range indices (a covariance).
        lines = self._src('oassun26.f')
        revint = '\n'.join(lines[844 - 1:845]).replace(' ', '')
        revran = '\n'.join(lines[624 - 1:625]).replace(' ', '')
        assert 'cff(index+iof,2)*conjg(cff(index+iof,2))' in revint
        assert 'cff(inr+iof,2)*conjg(cff(index+iof,2))' in revran

    @pytest.mark.requires_oases
    def test_option_r_reaches_revint_and_never_the_contour_branch(self):
        driver = self._src('unoass21.f')
        i = next(k for k, ln in enumerate(driver) if "OPT(I).EQ.'r'" in ln)
        branch = '\n'.join(driver[i:i + 8]).replace(' ', '')
        assert 'PLTL=.TRUE.' in branch
        assert 'CCONTU' not in branch
        j = next(k for k, ln in enumerate(driver) if 'CALL REVINT(' in ln)
        assert any('PLTL' in ln for ln in driver[j - 5:j])
        assert any('CALL PLTLOS(' in ln for ln in driver[j:j + 30])
        k = next(n for n, ln in enumerate(driver) if 'CALL REVRAN(' in ln)
        assert any('CCONTU' in ln for ln in driver[k - 5:k])

    @pytest.mark.requires_oases
    def test_the_contour_option_letter_is_the_capital_one(self):
        """OASES option letters are case-sensitive, and both cases exist here.
        The citation named lowercase ``'c'``, which sets ``ICONTU`` — the
        depth-integrand contours, a third output reaching neither REVRAN nor
        REVINT — so a reader who followed it and passed ``'c'`` expecting
        REVRAN would have got neither."""
        driver = self._src('unoass21.f')
        upper = next(k for k, ln in enumerate(driver) if 'OPT(I).EQ.\'C\'' in ln)
        lower = next(k for k, ln in enumerate(driver) if 'OPT(I).EQ.\'c\'' in ln)
        assert upper != lower
        assert any('CCONTU=.TRUE.' in ln.replace(' ', '')
                   for ln in driver[upper:upper + 4])
        assert any('ICONTU=.TRUE.' in ln.replace(' ', '')
                   for ln in driver[lower:lower + 4])
        # Neither ICONTU nor the lowercase letter reaches a reverberation
        # routine: REVRAN is guarded by CCONTU alone.
        k = next(n for n, ln in enumerate(driver) if 'CALL REVRAN(' in ln)
        assert not any('ICONTU' in ln for ln in driver[k - 5:k])

    def test_the_sentence_names_the_capital_letter_and_distinguishes_the_other(self):
        sentence = self._sentence()
        assert 'CCONTU' in sentence and 'ICONTU' in sentence
        assert 'case-sensitive' in sentence
        assert 'CAPITAL' in sentence or 'capital' in sentence

    @pytest.mark.requires_oases
    def test_line_876_of_the_source_is_an_include_block_not_a_conversion(self):
        # The first wrong address is real code, just not this code.
        old = '\n'.join(self._src('oassun26.f')[876 - 1:880])
        assert 'log10' not in old.lower()
        assert 'INCLUDE' in old

    def test_the_sentence_carries_the_address_and_rejects_both_wrong_ones(self):
        sentence = self._sentence()
        assert 'oassun26.f:853-858' in sentence
        assert '876-880' not in sentence
        # :633-638 may appear, but only as the block being ruled OUT — the
        # sentence has to say why, or the next reader re-derives the same
        # wrong answer from a content match.
        assert 'oassun26.f:633-638 is' in sentence
        assert 'REVINT' in sentence and 'REVRAN' in sentence
        assert 'CFFs' in sentence

    def test_the_core_and_model_citations_agree(self):
        # Two homes for one fact; they were allowed to drift once already.
        from uacpy.models.oases import OASS
        model_doc = inspect.getdoc(OASS._reverberation_field)
        assert 'oassun26.f:853-858' in model_doc
        assert 'oassun26.f:853-858' in self._sentence()


class TestTheResultsCoreStatesItsRegistryNotACopy:
    """The results core names its quantities from the registry: the metadata
    help lists every registered kind, and ``Field.max`` asks ``is_loss``
    rather than keeping its own pair of loss kinds."""

    def test_the_metadata_registry_holds_no_quantity_attribute(self):
        from uacpy.core.results._base import (_DOCUMENTED_METADATA,
                                              _UNIVERSAL_METADATA)
        quantity = {'kind', 'unit', 'coherent', 'reference',
                    'reference_unit'}
        assert not set(_UNIVERSAL_METADATA) & quantity
        assert not {key for (model, key) in _DOCUMENTED_METADATA
                    if model != 'OASN'} & quantity

    def test_field_max_follows_the_loss_registry(self, monkeypatch):
        from uacpy.core.results import quantities
        f = Field(data=np.array([[60.0, 70.0]]),
                  coords={'depth': np.array([10.0]),
                          'range': np.array([1.0, 2.0])},
                  kind='signal_excess')
        assert f.max().pinned['range'] == 2.0          # a level: more is more
        monkeypatch.setattr(quantities, 'LOSS_KINDS',
                            quantities.LOSS_KINDS + ('signal_excess',))
        assert f.max().pinned['range'] == 1.0          # a loss: less is louder


class TestArrayProductsCarryOneFrequencyPerSlice:
    """``Covariance`` and ``Replicas`` count frequencies off the array's
    axis 0, so a ``frequencies`` label axis of another length is refused at
    construction rather than answering a second count."""

    @staticmethod
    def _covariance(frequencies):
        from uacpy.core.results import Covariance
        return Covariance(covariance=np.tile(np.eye(2, dtype=complex),
                                             (2, 1, 1)),
                          frequencies=frequencies)

    @staticmethod
    def _replicas(frequencies):
        from uacpy.core.results import Replicas
        return Replicas(replicas=np.ones((2, 1, 1, 1, 2), complex),
                        candidates={'depth': [10.0], 'x': [0.0], 'y': [0.0]},
                        frequencies=frequencies)

    @pytest.mark.parametrize('build', ['_covariance', '_replicas'])
    def test_one_frequency_per_slice_is_accepted(self, build):
        result = getattr(self, build)([100.0, 200.0])
        assert result.n_frequencies == len(result.frequencies) == 2

    @pytest.mark.parametrize('build', ['_covariance', '_replicas'])
    @pytest.mark.parametrize('frequencies', [[100.0], [100.0, 200.0, 300.0]])
    def test_a_frequency_axis_of_another_length_is_refused(self, build,
                                                           frequencies):
        with pytest.raises(ConfigurationError,
                           match=rf"holds {len(frequencies)} value\(s\) but "
                                 r"the array's frequency axis \(axis 0\) "
                                 r"holds 2"):
            getattr(self, build)(frequencies)

    @pytest.mark.parametrize('build', ['_covariance', '_replicas'])
    def test_unset_frequencies_count_the_slices(self, build):
        result = getattr(self, build)(None)
        assert result.frequencies is None and result.n_frequencies == 2


class TestEveryResultTypeRoundTripsThroughADict:
    """``to_dict`` / ``from_dict`` on every result type, through
    ``np.savez`` (RES-15)."""

    @staticmethod
    def _round_trip(result, tmp_path):
        path = tmp_path / 'r.npz'
        np.savez(path, **result.to_dict())
        with np.load(path, allow_pickle=True) as loaded:
            return type(result).from_dict(dict(loaded))

    def test_rays(self, tmp_path):
        from uacpy.core.results import Rays
        rays = Rays(rays=[{'r': np.array([0.0, 10.0, 20.0]),
                           'z': np.array([5.0, 7.0, 9.0]), 'launch_angle': -3.0,
                           'n_top_bounces': 0, 'n_bot_bounces': 1},
                          {'r': np.array([0.0, 15.0]),
                           'z': np.array([5.0, 2.0]), 'launch_angle': 4.5,
                           'n_top_bounces': 1, 'n_bot_bounces': 0}],
                    is_eigen=True, receiver_depths=[9.0],
                    receiver_ranges=[20.0], model='Bellhop',
                    frequencies=[200.0])
        back = self._round_trip(rays, tmp_path)
        assert len(back.rays) == 2 and back.is_eigen
        for a, b in zip(rays.rays, back.rays):
            np.testing.assert_array_equal(a['r'], b['r'])
            np.testing.assert_array_equal(a['z'], b['z'])
            assert {k: a[k] for k in a if k not in 'rz'} == \
                {k: b[k] for k in b if k not in 'rz'}
        assert back.model == 'Bellhop' and back.f0 == 200.0

    def test_modes(self, tmp_path):
        from uacpy.core.results import MediaTable, Modes
        modes = Modes(k=np.array([0.8 + 1e-5j, 0.7 + 2e-5j]),
                      phi=np.arange(6.0).reshape(3, 2),
                      depths=np.array([0.0, 50.0, 100.0]),
                      group_velocity=np.array([1480.0, np.nan]),
                      model='Kraken', frequencies=[100.0],
                      media=MediaTable(water_density=1.0, tops=[0.0, 100.0],
                                       densities=[1.0, 1.8],
                                       bottom_depth=120.0,
                                       halfspace_density=2.0),
                      metadata={'title': 'round trip'})
        back = self._round_trip(modes, tmp_path)
        np.testing.assert_array_equal(back.k, modes.k)
        np.testing.assert_array_equal(back.phi, modes.phi)
        np.testing.assert_array_equal(back.group_velocity,
                                      modes.group_velocity)
        assert back.media == modes.media
        assert back.metadata == {'title': 'round trip'}

    def test_a_reflection_table(self, tmp_path):
        table = ReflectionCoefficient(angles=[10.0, 20.0], magnitude=[0.9, 0.8],
                                      phase=[0.3, 0.5], model='Bounce')
        back = self._round_trip(table, tmp_path)
        np.testing.assert_array_equal(back.coefficient, table.coefficient)
        assert back.phase_reference is PhaseReference.TRAVELLING_WAVE

    def test_a_covariance_and_its_replicas(self, tmp_path):
        from uacpy.core.results import Covariance, Replicas
        cov = Covariance(covariance=np.eye(2)[None] * (1 + 1j),
                         receiver_positions=np.zeros((2, 3)), model='OASN')
        back = self._round_trip(cov, tmp_path)
        np.testing.assert_array_equal(back.covariance, cov.covariance)
        np.testing.assert_array_equal(back.receiver_positions,
                                      cov.receiver_positions)
        rep = Replicas(replicas=np.ones((1, 2, 1, 1, 2), complex),
                       candidates={'depth': [10.0, 20.0], 'x': [1000.0],
                                   'y': [0.0]}, model='OASN')
        back = self._round_trip(rep, tmp_path)
        np.testing.assert_array_equal(back.replicas, rep.replicas)
        assert list(back.candidates) == ['depth', 'x', 'y']
        for name, values in rep.candidates.items():
            np.testing.assert_array_equal(back.candidates[name], values)
        assert back.receiver_positions is None


class TestTheRunModeIsOneOfTheRunModes:
    def test_a_misspelt_mode_is_refused(self):
        with pytest.raises(ConfigurationError, match='cohernt_tl'):
            _field(run_mode='cohernt_tl')
        assert _field(run_mode='coherent_tl').run_mode == 'coherent_tl'

    def test_every_member_and_its_value_is_accepted_and_the_refusal_lists_them(
            self):
        """The check reads the enum itself: each member and each member's
        value string is accepted, and the refusal lists every value in
        member order."""
        from uacpy.core.run_settings import RunMode
        for mode in RunMode:
            assert _field(run_mode=mode).run_mode is mode
            assert _field(run_mode=mode.value).run_mode == mode
        values = [m.value for m in RunMode]
        with pytest.raises(ConfigurationError,
                           match="run_mode='Coherent_TL'") as excinfo:
            _field(run_mode='Coherent_TL')
        assert str(excinfo.value).endswith(f"one of {values}.")


class TestEveryResultCarriesTheSettingsItRanWith:
    """``result.run_settings`` is read-only, inherited by every derived
    result, and travels through ``to_dict`` (as plain types) and
    ``to_xarray`` (as the one-line summary)."""

    @staticmethod
    def _settings():
        from uacpy.core.receiver import Receiver
        env = Environment(name='triple', bathymetry=100.0, ssp=1500.0)
        return uacpy.Kraken().run_settings(
            env, Source(depths=25.0, frequencies=200.0),
            Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])))

    def _field(self):
        f = _field(data=np.full((4, 3), 60.0))
        f._run_settings = self._settings()
        return f

    def test_it_is_read_only_and_a_slice_inherits_it(self):
        f = self._field()
        with pytest.raises(AttributeError, match='has no setter'):
            f.run_settings = None
        assert f.at(depth=10.0).run_settings is f.run_settings
        assert f.to_dB().run_settings is f.run_settings
        stack = ResultStack([f, f], [1.0, 2.0])
        assert stack.run_settings is f.run_settings

    @pytest.mark.parametrize('synthesise', [
        lambda H: H.synthesize_time_series(
            np.sin(2 * np.pi * 200.0 * np.arange(0.0, 0.02, 1 / 2000.0)),
            2000.0, t_start=0.0),
        lambda H: H.to_time_trace(depth=10.0, range=300.0, t_start=0.0),
    ], ids=['synthesize_time_series', 'to_time_trace'])
    def test_a_trace_synthesised_from_a_run_carries_its_settings(
            self, synthesise):
        H = _two_path_grid()
        H._run_settings = self._settings()
        assert synthesise(H).run_settings is H.run_settings

    def test_the_transfer_function_of_arrivals_carries_their_settings(self):
        a = uacpy.Arrivals(
            arrivals=[{'delay': 0.2, 'amplitude': 1.0, 'phase': 0.0}],
            receiver_depths=[50.0], receiver_ranges=[1000.0])
        a._run_settings = self._settings()
        H = a.transfer_function(np.arange(100.0, 110.0))
        assert H.run_settings is a.run_settings

    def test_it_round_trips_through_a_dict_and_a_file(self, tmp_path):
        f = self._field()
        d = f.to_dict()
        assert d['run_settings'] == f.run_settings.to_dict()
        np.savez(tmp_path / 'f.npz', **d)
        with np.load(tmp_path / 'f.npz', allow_pickle=True) as loaded:
            back = Field.from_dict(dict(loaded))
        assert back.run_settings.to_dict() == f.run_settings.to_dict()

    def test_xarray_carries_the_one_line_summary(self):
        pytest.importorskip('xarray')
        f = self._field()
        da = f.to_xarray()
        assert da.attrs['run_settings_summary'] == f.run_settings.summary()
        assert '\n' not in da.attrs['run_settings_summary']
        back = Field.from_xarray(da)
        assert back.metadata['run_settings_summary'] == \
            f.run_settings.summary()

    def test_xarray_and_netcdf_carry_the_whole_record(self, tmp_path):
        pytest.importorskip('xarray')
        f = self._field()
        back = Field.from_xarray(f.to_xarray())
        assert back.run_settings.to_dict() == f.run_settings.to_dict()
        f.to_netcdf(tmp_path / 'f.nc')
        back = Field.from_netcdf(tmp_path / 'f.nc')
        assert back.run_settings.to_dict() == f.run_settings.to_dict()
        assert isinstance(uacpy.Kraken.from_run_settings(back.run_settings),
                          uacpy.Kraken)

    def test_a_record_read_through_the_dict_route_comes_back(self):
        pytest.importorskip('xarray')
        rays = uacpy.Rays(rays=[{'r': np.array([0.0, 100.0]),
                                 'z': np.array([25.0, 30.0])}])
        rays._run_settings = self._settings()
        back = uacpy.Rays.from_xarray(rays.to_xarray())
        assert back.run_settings.to_dict() == rays.run_settings.to_dict()

    def test_a_result_built_by_hand_writes_no_record(self):
        pytest.importorskip('xarray')
        f = _field(data=np.full((4, 3), 60.0))
        da = f.to_xarray()
        assert 'run_settings' not in da.attrs
        assert Field.from_xarray(da).run_settings is None


class TestAResultCarriesItsComponents:
    """``components``: the results a result was built from, by a fixed set of
    names, read-only, the objects themselves; a same-quantity Field
    derivation keeps them, a change of quantity drops them."""

    @staticmethod
    def _arrivals():
        from uacpy.core.results import Arrivals
        return Arrivals(arrivals=[], receiver_depths=[10.0],
                        receiver_ranges=[100.0])

    def _field(self, **kwargs):
        from uacpy.core.results import Field
        return Field(data=np.ones((2, 3), complex),
                     coords={'depth': np.array([1.0, 2.0]),
                             'range': np.array([1.0, 2.0, 3.0])},
                     **kwargs)

    def test_a_result_without_components_has_an_empty_mapping(self):
        assert dict(self._field().components) == {}

    def test_the_components_are_the_objects_given_and_read_only(self):
        arr = self._arrivals()
        f = self._field(components={'arrivals': arr})
        assert f.components['arrivals'] is arr
        with pytest.raises(TypeError, match='does not support item'):
            f.components['bounce'] = arr

    def test_an_unknown_name_is_refused(self):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match="'arrival_field'"):
            self._field(components={'arrival_field': self._arrivals()})

    def test_a_value_that_is_not_a_result_is_refused(self):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='not results'):
            self._field(components={'arrivals': [1, 2]})

    def test_a_same_quantity_derivation_keeps_them(self):
        arr = self._arrivals()
        f = self._field(components={'arrivals': arr})
        for derived in (f.replace(data=f.data * 2), f.window(range=(1.5, 3.0)),
                        f.isel(depth=0), f.to_dB()):
            assert derived.components['arrivals'] is arr

    def test_a_change_of_quantity_drops_them(self):
        f = self._field(components={'arrivals': self._arrivals()})
        assert dict(f.replace(kind='level', unit='dB',
                              data=np.ones((2, 3))).components) == {}

    def test_the_identity_keywords_do_not_carry_them(self):
        f = self._field(components={'arrivals': self._arrivals()})
        assert 'components' not in f.id_kwargs()
