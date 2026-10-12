"""Every engine's capability surface, held to its saved record.

What each registered engine declares (``spec``: run modes, capability flags,
source geometries, collapse defaults, ``EngineTraits``) and what a default
instance resolves from it (its run modes, every ``_supports_*`` flag, its
source geometries, its collapse policy, its default run mode, what a
multi-depth ``Source`` meets in that mode) is saved in
``tests/data/engine_specs.json``. ``python -m uacpy.tests._save_engine_specs``
writes it; the tests below compare the live values with it field by field,
so a capability that changes fails naming the engine and the field, and an
intended change is recorded by rewriting the engine's record.

Declaring a flag is a promise: ``_project_environment`` then leaves that
feature in the env and warns about nothing, so the model's deck writer has to
carry it — nothing downstream re-checks. A flag held here without a
companion test that drives the writer is therefore only half the contract.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import uacpy

from uacpy.models.bellhop import Bellhop
from uacpy.models.scooter import Scooter
from uacpy.models.sparc import SPARC
from uacpy.models.bounce import Bounce
from uacpy.models.ram import RAM
from uacpy.models._registry import ENGINES
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, UnsupportedFeatureError,
)
from uacpy.core.source import VALID_SOURCE_TYPES
from uacpy.tests._save_engine_specs import (
    default_engine, instance_record, spec_record,
)
from uacpy.tests.conftest import build_engine, engine_entry, engine_params


#: Every registered engine's saved record (module docstring).
_SAVED_SPECS = Path(__file__).parent / 'data' / 'engine_specs.json'


def _saved_records() -> dict:
    """The saved records, by registry key."""
    return json.loads(_SAVED_SPECS.read_text(encoding='utf-8'))


def _field_differences(saved, live, path=''):
    """``(field, saved value, live value)`` for every value where the two
    records differ, the field a dotted path (``traits.consumes_run_t_start``);
    a field on one side only reads ``'<absent>'`` on the other."""
    if isinstance(saved, dict) and isinstance(live, dict):
        out = []
        for key in sorted(set(saved) | set(live)):
            field = f'{path}.{key}' if path else key
            if key not in saved:
                out.append((field, '<absent>', live[key]))
            elif key not in live:
                out.append((field, saved[key], '<absent>'))
            else:
                out.extend(_field_differences(saved[key], live[key], field))
        return out
    return [] if saved == live else [(path, saved, live)]


def _assert_matches_saved(key, part, live):
    """Fail naming the engine and every field of ``part`` (``'spec'`` or
    ``'instance'``) where ``live`` differs from engine ``key``'s saved
    record, with the command that rewrites it."""
    saved_by_key = _saved_records()
    if key not in saved_by_key:
        pytest.fail(f"no saved record for {key!r} in "
                    f"tests/data/engine_specs.json: run python -m "
                    f"uacpy.tests._save_engine_specs {key} "
                    f"(docs/DEV.md section 3, step 5)")
    saved = saved_by_key[key]
    live = json.loads(json.dumps(live, sort_keys=True))
    differences = _field_differences(saved[part], live)
    assert not differences, (
        f"{saved['class']} differs from its saved record "
        f"(tests/data/engine_specs.json):\n"
        + "\n".join(f"  {saved['class']}: {part}.{field}: saved {old!r}, "
                    f"live {new!r}" for field, old, new in differences)
        + f"\nIf the change is meant, rewrite the record: python -m "
          f"uacpy.tests._save_engine_specs {key}")


#: One parameter per registered engine, by registry key, its class name as
#: the id; the instance tests carry the markers of the installs the entry
#: declares, since every constructor resolves its binary.
_SPEC_KEYS = [pytest.param(key, id=entry.class_name)
              for key, entry in ENGINES.items()]
_INSTANCE_KEYS = [
    pytest.param(key, id=entry.class_name,
                 marks=[getattr(pytest.mark, f'requires_{install}')
                        for install in entry.requires])
    for key, entry in ENGINES.items()]


def test_the_saved_records_cover_the_registered_engines():
    """A registered engine with no saved record would be compared with
    nothing, and a record of an engine the registry no longer holds is a
    stale one."""
    saved = set(_saved_records())
    missing = sorted(set(ENGINES) - saved)
    stale = sorted(saved - set(ENGINES))
    assert not missing and not stale, (
        f"tests/data/engine_specs.json has no record for {missing} and stale "
        f"records for {stale}: run python -m uacpy.tests._save_engine_specs "
        f"(docs/DEV.md section 3, step 5)")


@pytest.mark.parametrize('key', _SPEC_KEYS)
def test_an_engine_declares_its_saved_spec(key):
    """``spec`` — run modes in order, capability flags, source geometries,
    collapse defaults and every ``EngineTraits`` field — equals the saved
    one. Read off the class: nothing is constructed."""
    cls = ENGINES[key].load()
    assert _saved_records()[key]['class'] == cls.__name__
    _assert_matches_saved(key, 'spec', spec_record(cls))


@pytest.mark.parametrize('key', _INSTANCE_KEYS)
def test_a_default_engine_resolves_its_saved_capabilities(key):
    """What a default instance resolves — run modes, every capability flag
    through ``supports_feature``, source geometries, collapse policy, default
    run mode, and what a two-depth ``Source`` meets in that mode, asked of
    ``validate_inputs`` — equals the saved record. Read off the instance,
    because an engine may resolve a flag from its constructor arguments
    (Bellhop carries ``range_dependent_ssp`` for the default ``interp_ssp``
    without declaring it)."""
    try:
        model = default_engine(key)
    except ExecutableNotFoundError:
        pytest.skip(f"{ENGINES[key].class_name} binary not available")
    _assert_matches_saved(key, 'instance', instance_record(key, model))


_MODEL_PARAMS = engine_params(value='name')


def _reference_environment(model_name=None):
    """An isovelocity 200 m guide over the default half-space, or over the
    seabed the engine's registry entry names (``EngineEntry.example_bottom``):
    a rigid floor for SPARC, whose deck carries only vacuum / rigid seabeds,
    and a rough one for OASSP and OASS, which scatter from it."""
    kw = {}
    bottom = dict(engine_entry(model_name).example_bottom) if model_name else {}
    if bottom:
        kw['bottom'] = uacpy.BoundaryProperties(**bottom)
    return uacpy.Environment(
        bathymetry=200.0,
        ssp=uacpy.SoundSpeedProfile(depths=[0, 200], sound_speed=[1500, 1500]),
        **kw,
    )


def _reference_receiver():
    return uacpy.Receiver(depths=100.0, ranges=np.linspace(100, 2000, 20))


def test_the_two_reflection_models_accept_the_same_source_types():
    """``RunMode.REFLECTION`` is answered by Bounce and OASR, and neither
    reads source geometry — Bounce says so in its spec comment and OASR's
    deck writer reads only ``source.frequencies``. A ``Source`` that works on
    one must work on the other, or reusing it across models breaks for a
    reason no engine has."""
    from uacpy.models.oases import OASR as _OASR
    assert set(_OASR.spec.source_types) == set(Bounce.spec.source_types)


@pytest.mark.requires_oases
@pytest.mark.parametrize('source_type', ['point', 'line', 'scaled'])
def test_oasr_returns_the_same_coefficient_for_every_source_type(source_type):
    """And the acceptance is honest: the answer does not move, so widening
    the declaration cannot have changed a number."""
    try:
        from uacpy.models.oases import OASR as _OASR
        model = _OASR(verbose=False)
    except ExecutableNotFoundError:
        pytest.skip("OASR binary not available")
    env = uacpy.Environment(
        bathymetry=100.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(sound_speed=1700.0, density=1.7,
                                        attenuation=0.5))
    receiver = uacpy.Receiver(depths=[50.0], ranges=[1000.0])
    reference = model.run(
        env, uacpy.Source(depths=25.0, frequencies=100.0), receiver)
    result = model.run(
        env,
        uacpy.Source(depths=25.0, frequencies=100.0, source_type=source_type),
        receiver)
    assert np.any(np.asarray(reference.magnitude) != 0.0), (
        "the reference coefficient is identically zero — this fixture cannot "
        "tell an unchanged answer from an absent one")
    assert np.array_equal(np.asarray(result.magnitude), np.asarray(reference.magnitude))


def test_every_saved_source_type_is_valid():
    """The geometries each record names — declared and resolved — are ones a
    ``Source`` accepts."""
    for record in _saved_records().values():
        assert set(record['spec']['source_types']) <= VALID_SOURCE_TYPES
        assert set(record['instance']['source_types']) <= VALID_SOURCE_TYPES


@pytest.mark.requires_binary
def test_sparc_honours_no_source_geometry():
    # A source geometry is a weighting inside the wavenumber->range Hankel
    # transform, and only the snapshot mode runs one (``SPARC._to_result``
    # hands ``source_type`` to ``GreensFunction.snapshot_to_time_field``). The
    # default ``output_mode='R'`` and ``'D'`` are range- / depth-native and
    # never reach it, so they honour no geometry beyond a point source.
    try:
        assert set(SPARC()._supported_source_types) == {'point'}
    except ExecutableNotFoundError:
        pytest.skip("SPARC binary not available")


@pytest.mark.requires_binary
def test_unsupported_source_type_is_rejected():
    try:
        m = RAM()
    except ExecutableNotFoundError:
        pytest.skip("RAM binary not available")
    with pytest.raises(UnsupportedFeatureError, match="source_type"):
        m.validate_inputs(
            _reference_environment(),
            uacpy.Source(depths=50, frequencies=100, source_type='line'),
            _reference_receiver(),
        )


@pytest.mark.requires_binary
def test_unsupported_beam_pattern_is_rejected():
    try:
        m = Scooter()
    except ExecutableNotFoundError:
        pytest.skip("Scooter binary not available")
    pat = np.array([[-90.0, -20.0], [90.0, 0.0]])
    with pytest.raises(ConfigurationError, match="beam pattern"):
        m.validate_inputs(
            _reference_environment(),
            uacpy.Source(depths=50, frequencies=100, beam_pattern=pat),
            _reference_receiver(),
        )


class TestPublicEnvShapeAccessors:
    """``supported_features`` / ``supports_feature`` are the env-shape twins
    of ``supported_modes`` / ``supports_mode``.

    They read the *instance* flags, which is the only place the answer is
    right for a model that resolves a flag from its constructor arguments —
    ``Bellhop`` declares no ``range_dependent_ssp`` in ``spec.supports`` and
    carries it on every instance whose ``interp_ssp`` can express a 2-D
    profile.
    """

    @pytest.mark.parametrize('model_name', _MODEL_PARAMS)
    def test_the_accessors_agree_with_the_private_flags(self, model_name):
        try:
            m = build_engine(model_name)
        except ExecutableNotFoundError:
            pytest.skip(f"{model_name} binary not available")
        from uacpy.models._spec import _CAPABILITY_FLAGS
        for name in _CAPABILITY_FLAGS:
            assert m.supports_feature(name) is bool(
                getattr(m, f'_supports_{name}'))
        assert m.supported_features == sorted(
            n for n in _CAPABILITY_FLAGS if getattr(m, f'_supports_{n}'))

    def test_an_unknown_feature_name_raises_rather_than_answering_no(self):
        """A typo answering ``False`` reads as a real "this model cannot do
        it" — the failure this accessor exists to prevent."""
        try:
            m = Bellhop()
        except ExecutableNotFoundError:
            pytest.skip("Bellhop binary not available")
        with pytest.raises(ConfigurationError, match='unknown capability'):
            m.supports_feature('range_dependant_ssp')

    def test_a_known_name_next_to_the_typo_answers(self):
        """The other side of the same check."""
        try:
            m = Bellhop()
        except ExecutableNotFoundError:
            pytest.skip("Bellhop binary not available")
        assert m.supports_feature('range_dependent_ssp') is True

    def test_the_accessor_is_instance_correct_where_the_spec_is_not(self):
        """The case that motivates reading the instance: ``interp_ssp``
        decides, and ``Bellhop.spec.supports`` cannot know it."""
        try:
            quad = Bellhop(interp_ssp='quad')
            linear = Bellhop(interp_ssp='linear')
        except ExecutableNotFoundError:
            pytest.skip("Bellhop binary not available")
        assert 'range_dependent_ssp' not in Bellhop.spec.supports
        assert quad.supports_feature('range_dependent_ssp') is True
        assert linear.supports_feature('range_dependent_ssp') is False


@pytest.mark.requires_binary
def test_rough_surface_is_dropped_for_solvers_that_reject_it():
    """A rough surface must be collapsed with a warning, not passed through.

    Surface roughness reaches the water column's mesh line (sigma(1)); handing
    it to SPARC would trip its 'Rough interfaces not allowed' ERROUT.
    """
    import warnings
    env = _reference_environment()
    env.surface.roughness = 2.0
    try:
        m = build_engine('SPARC')
    except ExecutableNotFoundError:
        pytest.skip("SPARC binary not available")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        projected = m._project_environment(env)
    assert projected.surface.roughness == 0.0
    assert [x for x in w if 'rough sea surface' in str(x.message)]
    # The caller's environment must not be mutated in place.
    assert env.surface.roughness == 2.0


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_the_multi_source_depth_flag_is_read_off_the_native_modes(model_name):
    """``_supports_multi_source_depth`` is True exactly for the models that
    run a source-depth *grid* in one binary call, the modes being
    ``spec.traits.native_multi_depth_modes``; the flag is the whole-model
    statement the capability matrix in docs/models/README.md prints, and it
    is never declared in ``spec.supports``. Every other model reads
    ``False``: in a field mode ``PropagationModel.run`` loops over the depths
    and stacks the slabs, and in any other mode ``_validate_geometry``
    refuses a multi-depth Source with 'single source depth' (pinned in
    test_input_validation.py). Bounce also reads ``False``, but its geometry
    validation is a no-op so nothing enforces it.
    """
    try:
        m = build_engine(model_name)
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    assert m._supports_multi_source_depth is bool(
        type(m).spec.traits.native_multi_depth_modes)
    assert 'multi_source_depth' not in type(m).spec.supports


def test_multi_source_depth_declared_in_the_spec_is_refused():
    """The flag is read off ``spec.traits.native_multi_depth_modes``; declaring it in
    ``spec.supports`` as well would give one question two answers, so the
    declaration is refused at construction."""
    from uacpy.core.run_settings import RunMode
    from uacpy.models.base import PropagationModel
    from uacpy.models._spec import ModelSpec

    def _never_launched(self, *args):
        raise AssertionError('refused at construction')

    class _Declares(PropagationModel):
        spec = ModelSpec(modes=(RunMode.COHERENT_TL,),
                         supports={'multi_source_depth'})
        provenance_id = 'acoustics_toolbox'
        _write_input = _launch = _read_output = _to_result = _never_launched

    with pytest.raises(ConfigurationError, match='multi_source_depth'):
        _Declares()


@pytest.mark.requires_binary
class TestRoughBottomCapability:
    """``_supports_rough_bottom`` decides whether ``_project_environment``
    keeps the seabed sigma or drops it with a warning — symmetric with
    rough_surface, so a model that cannot deliver it never silently discards
    the caller's input.

    These cases stop at the projection; they do not show whether the declaring
    model actually consumes the value. ``TestScooterRoughnessReachesTheSolver``
    below closes that half for Scooter by running the binary.
    """

    @staticmethod
    def _env(sigma):
        import uacpy
        return uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0,
                density=1.8, attenuation=0.5, roughness=sigma))

    @pytest.mark.parametrize('model_name', ['Bellhop', 'RAM', 'Scooter'])
    def test_models_that_ignore_it_warn_and_collapse(self, model_name):
        import uacpy
        m = getattr(uacpy, model_name)(verbose=False)
        with pytest.warns(UserWarning, match="seabed interfacial roughness"):
            projected = m._project_environment(self._env(3.0))
        assert projected.bottom.columns[0].halfspace.roughness == 0.0

    @pytest.mark.parametrize('model_name', ['Kraken'])
    def test_models_that_honour_it_keep_it(self, model_name):
        # Projection only: the assertion is that the sigma survives into the
        # env handed to the writer, not that the solver reads it back.
        import uacpy
        import warnings as _w
        m = getattr(uacpy, model_name)(verbose=False)
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            projected = m._project_environment(self._env(3.0))
        assert projected.bottom.columns[0].halfspace.roughness == 3.0

    def test_collapse_rebuilds_rather_than_shadowing(self):
        """Surface.roughness is served by __getattr__ delegation, so a plain
        assignment would shadow it while properties[] would keep the previous
        value."""
        import uacpy
        env = uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                            sound_speed=1700.0, density=1.8,
                                            attenuation=0.5),
            surface=uacpy.BoundaryProperties(acoustic_type='half-space',
                                             sound_speed=1600.0, density=0.9,
                                             attenuation=0.0, roughness=3.0))
        m = uacpy.Bellhop(verbose=False)
        with pytest.warns(UserWarning, match="rough sea surface"):
            projected = m._project_environment(env)
        assert projected.surface.roughness == 0.0
        assert projected.surface.nodes[0].roughness == 0.0, (
            "collapse only shadowed the delegating attribute")
        assert projected.surface.at(range=0.0).roughness == 0.0

    @pytest.mark.requires_binary
    def test_scooter_layer_roughness_is_collapsed_not_run(self):
        """A *layer* sigma is the fatal case, not merely the inert one.

        ``write_layer_sections`` writes it onto the layer's mesh line, i.e.
        ``SSP%sigma(2:NMedia)`` — the exact range ``Scooter/scooter.f90:63``
        stops the run on. Without the projection this raises
        ``ModelExecutionError('Rough interfaces not allowed')``.
        """
        import uacpy
        try:
            m = uacpy.Scooter(verbose=False)
        except ExecutableNotFoundError:
            pytest.skip("Scooter binary not available")
        env = uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.Bottom([uacpy.SeabedColumn(
                layers=[uacpy.SedimentLayer(
                    thickness=20.0, sound_speed=1600.0, density=1.7,
                    attenuation=0.3, roughness=1.0)],
                halfspace=uacpy.BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1800.0,
                    density=2.0, attenuation=0.5))]))
        with pytest.warns(UserWarning, match="seabed interfacial roughness"):
            result = m.run(
                env,
                uacpy.Source(depths=25.0, frequencies=100.0),
                uacpy.Receiver(depths=50.0,
                               ranges=np.linspace(500, 3000, 11)),
            )
        assert np.all(np.isfinite(result.dB))


@pytest.mark.requires_binary
class TestScooterRoughnessReachesTheSolver:
    """End-to-end: does a declared roughness actually move Scooter's answer?

    ``SSP%sigma(1)`` enters the solve only through the vacuum branch of
    ``Scooter/scooter.f90:309`` (``g = -i·sqrt(omega2/cInside² − x)·
    sigma(1)²``), reached from ``:635`` via ``BCImpedance(x, 'TOP', …)``. That
    makes ``rough_surface`` real for a pressure-release surface and inert for
    every other top boundary condition, which is why ``Scooter`` drops it in
    the latter case rather than writing a value the run ignores.
    """

    @staticmethod
    def _run(roughness):
        import uacpy
        import warnings as _w
        env = uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0,
                density=1.8, attenuation=0.5))
        env.surface.roughness = roughness
        m = uacpy.Scooter(verbose=False)
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            return m.run(
                env,
                uacpy.Source(depths=25.0, frequencies=100.0),
                uacpy.Receiver(depths=np.linspace(5, 95, 19),
                               ranges=np.linspace(500, 5000, 46)),
            )

    def test_vacuum_surface_roughness_changes_the_field(self):
        try:
            smooth = self._run(0.0)
        except ExecutableNotFoundError:
            pytest.skip("Scooter binary not available")
        rough = self._run(2.0)
        assert np.nanmax(np.abs(smooth.dB - rough.dB)) > 1.0

    def test_rigid_surface_roughness_is_dropped_with_a_warning(self):
        import uacpy
        try:
            m = uacpy.Scooter(verbose=False)
        except ExecutableNotFoundError:
            pytest.skip("Scooter binary not available")
        env = uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0,
                density=1.8, attenuation=0.5),
            surface=uacpy.BoundaryProperties(acoustic_type='rigid',
                                             roughness=2.0))
        with pytest.warns(UserWarning, match="pressure-release"):
            projected = m._project_environment(env)
        assert projected.surface.roughness == 0.0
