"""Capability-flag harmonization tests.

Locks in the per-model `_supports_*` matrix. If a model gains or loses
support for an Environment feature, the change must come with an update
to this test so the public capability surface stays explicit.

Declaring a flag is a promise: ``_project_environment`` then leaves that
feature in the env and warns about nothing, so the model's deck writer has to
carry it — nothing downstream re-checks. A flag asserted here without a
companion test that drives the writer is therefore only half the contract.
"""

import numpy as np
import pytest

import uacpy

from uacpy.models.bellhop import Bellhop
from uacpy.models.scooter import Scooter
from uacpy.models.sparc import SPARC
from uacpy.models.bounce import Bounce
from uacpy.models.ram import RAM
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, UnsupportedFeatureError,
)
from uacpy.core.source import VALID_SOURCE_TYPES
from uacpy.tests.conftest import (build_engine, engine_entry, engine_names,
                                  engine_params)


_FEATURES = (
    'altimetry',
    'range_dependent_bathymetry',
    'range_dependent_ssp',
    'range_dependent_bottom',
    'layered_bottom',
    'elastic_media',
)


# Expected flags by feature, per engine. Each test builds its engine with
# ``build_engine`` behind the markers ``engine_params`` attaches, because
# every constructor resolves its binary.
_EXPECTED = {
    'Bellhop':
        {'altimetry': True, 'range_dependent_bathymetry': True,
         'range_dependent_ssp': True,
         'range_dependent_bottom': True, 'layered_bottom': False,
         'elastic_media': True},
    'Kraken':
        {'altimetry': False, 'range_dependent_bathymetry': True,
         'range_dependent_ssp': True,
         'range_dependent_bottom': True, 'layered_bottom': True,
         'elastic_media': True},
    'Scooter':
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    'SPARC':
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': False},
    'Bounce':
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    'OAST':
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    'OASN':
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    'OASR':
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    'OASP':
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    # OASSP and OASS read the layered stack OASP solves the mean field on
    # (INENVI is shared verbatim between the binaries), so their axes are
    # OASP's.
    'OASSP':
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    'OASS':
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    'RAM':
        {'altimetry': True, 'range_dependent_bathymetry': True,
         'range_dependent_ssp': True,
         'range_dependent_bottom': True, 'layered_bottom': True,
         'elastic_media': True},
}


_MODEL_PARAMS = engine_params(value='name')


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
@pytest.mark.parametrize('feature', _FEATURES)
def test_capability_flag(model_name, feature):
    expected = _EXPECTED[model_name]
    try:
        m = build_engine(model_name)
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    flag = getattr(m, f'_supports_{feature}')
    assert flag is expected[feature], (
        f"{model_name}._supports_{feature} = {flag}, "
        f"expected {expected[feature]}"
    )


# (source geometries the model honours, whether it reads a .sbp beam pattern).
# Every entry is read off a default-constructed model. SPARC's is the only
# instance-dependent one: ``SPARC(output_mode='S')`` widens it to all three
# (``SPARC.__init__``), so the ``{'point'}`` below pins the default
# ``output_mode='R'`` and nothing here covers the snapshot mode.
_EXPECTED_SOURCE_TYPES = {
    'Bellhop': ({'point', 'line'}, True),
    'Kraken':  ({'point', 'line', 'scaled'}, True),
    'Scooter': ({'point', 'line', 'scaled'}, False),
    'SPARC':   ({'point'}, False),
    'Bounce':  ({'point', 'line', 'scaled'}, False),
    # OAST runs a line source in its plane geometry, option 'P' (OASES-11).
    'OAST':    ({'point', 'line'}, False),
    # OASN has no line-source geometry: its 'P' selects noise-intensity plots
    # (unoasn22.f:655).
    'OASN':    ({'point'}, False),
    # A plane-wave reflection coefficient does not depend on source geometry,
    # and OASR's deck writer reads only source.frequencies — same reasoning
    # as Bounce, same run mode.
    'OASR':    ({'point', 'line', 'scaled'}, False),
    # OASP runs a line source in its plane geometry, option 'P' (decision A4).
    'OASP':    ({'point', 'line'}, False),
    # OASSP and OASS run both of their decks in plane geometry for a line
    # Source, option 'P' (unoassp30.f:959-961, unoass21.f:654-656).
    'OASSP':   ({'point', 'line'}, False),
    'OASS':    ({'point', 'line'}, False),
    'RAM':     ({'point'}, False),
}


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


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_source_capability_matrix(model_name):
    """Locks the per-model source-geometry / beam-pattern surface."""
    try:
        m = build_engine(model_name)
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    types, pattern = _EXPECTED_SOURCE_TYPES[model_name]
    assert set(m._supported_source_types) == types
    assert m._supports_source_beam_pattern is pattern


def test_the_two_reflection_models_accept_the_same_source_types():
    """``RunMode.REFLECTION`` is answered by Bounce and OASR, and neither
    reads source geometry — Bounce says so in its spec comment and OASR's
    deck writer reads only ``source.frequencies``. A ``Source`` that works on
    one must work on the other, or reusing it across models breaks for a
    reason no engine has."""
    from uacpy.models.oases import OASR as _OASR
    assert (set(_EXPECTED_SOURCE_TYPES['OASR'][0])
            == set(_EXPECTED_SOURCE_TYPES['Bounce'][0]))
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


def test_every_table_covers_the_registered_engines():
    """A table missing an engine would skip that engine's row, and a row
    for an engine the registry does not hold is a stale one."""
    for name, table in (
            ('_EXPECTED', _EXPECTED),
            ('_EXPECTED_SOURCE_TYPES', _EXPECTED_SOURCE_TYPES),
            ('_EXPECTED_ROUGH_SURFACE', _EXPECTED_ROUGH_SURFACE),
            ('_EXPECTED_MULTI_SOURCE_DEPTH', _EXPECTED_MULTI_SOURCE_DEPTH),
            ('_EXPECTED_ROUGH_BOTTOM', _EXPECTED_ROUGH_BOTTOM),
            ('_VOLUME_ATTENUATION', _VOLUME_ATTENUATION),
            ('_MULTI_DEPTH_IN_DEFAULT_MODE', _MULTI_DEPTH_IN_DEFAULT_MODE)):
        missing = sorted(engine_names() - set(table))
        stale = sorted(set(table) - engine_names())
        assert not missing and not stale, (
            f"{name} in test_capability_flags.py has no row for {missing} "
            f"and stale rows for {stale}: every registered engine needs one "
            f"(docs/DEV.md section 3, step 5)")


def test_every_declared_source_type_is_valid():
    for types, _ in _EXPECTED_SOURCE_TYPES.values():
        assert types <= VALID_SOURCE_TYPES


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


_EXPECTED_ROUGH_SURFACE = {
    'Bellhop': False, 'Kraken': True, 'Scooter': True, 'SPARC': False,
    'Bounce': False, 'OAST': True, 'OASN': True,
    # OASR's deck has no sea surface: its layer 1 is the water half-space
    # the plane wave arrives through, whose RG INENVI discards
    # (oaseun31.f:377), so surface roughness is collapsed with a warning.
    'OASR': False,
    'OASP': True, 'OASSP': True, 'OASS': True, 'RAM': False,
}


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_rough_surface_capability_matrix(model_name):
    """Only solvers that accept a non-zero SSP%sigma may declare it.

    ``Scooter/sparc.f90:177`` ERROUTs on any non-zero ``SSP%sigma(1:NMedia)``
    and ``Kraken/bounce.f90:104`` on a rough elastic interface, so those must
    not receive ``env.surface.roughness``; Kraken and Scooter consume it
    (Kraken via the Kuperman-Ingenito perturbation, on any top boundary whose
    ``Kraken/kraken.f90:850-867`` branch leaves ``rho1`` non-zero — 'A', 'V'
    and 'R'; Scooter at ``Scooter/scooter.f90:309``, where ``SSP%sigma(1)``
    enters the vacuum-boundary impedance). The OASES family reads it as column 7 (RG) of
    each layer record (``oases/src/oaseun31.f:54``,
    ``oases/doc/oast.tex:48``) — except OASR, whose deck has no sea surface
    at all (see the matrix entry).
    """
    try:
        m = build_engine(model_name)
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    assert m._supports_rough_surface is _EXPECTED_ROUGH_SURFACE[model_name]


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


_EXPECTED_MULTI_SOURCE_DEPTH = {
    'Bellhop': True, 'Kraken': True, 'Scooter': True, 'SPARC': False,
    'Bounce': False, 'OAST': False, 'OASN': False, 'OASR': False,
    'OASP': False, 'OASSP': False, 'OASS': False, 'RAM': False,
}


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_multi_source_depth_capability_matrix(model_name):
    """``_supports_multi_source_depth`` is True for the models that run a
    source-depth *grid* in one binary call: Bellhop (in
    every mode it stacks), Kraken (in its two TL modes) and Scooter (in
    ``COHERENT_TL``). Which modes
    those are is ``spec.traits.native_multi_depth_modes``; this flag is the whole-model
    statement the capability matrix in docs/models/README.md prints. Every
    other model reads ``False``: in a field mode ``PropagationModel.run``
    loops over the depths and stacks the slabs, and in any other mode
    ``_validate_geometry`` refuses a multi-depth Source with 'single source
    depth' (pinned in test_input_validation.py). Bounce also reads
    ``False``, but its geometry validation is a no-op so nothing enforces
    it.
    """
    try:
        m = build_engine(model_name)
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    assert (m._supports_multi_source_depth
            is _EXPECTED_MULTI_SOURCE_DEPTH[model_name])
    assert m._supports_multi_source_depth is bool(
        type(m).spec.traits.native_multi_depth_modes)
    assert 'multi_source_depth' not in type(m).spec.supports


def test_multi_source_depth_declared_in_the_spec_is_refused():
    """The flag is read off ``spec.traits.native_multi_depth_modes``; declaring it in
    ``spec.supports`` as well would give one question two answers, so the
    declaration is refused at construction (ARCH-11)."""
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


#: What a two-depth Source meets in each engine's default run mode:
#: ``'field'`` — the default mode is a field mode, which stacks the depths;
#: ``'ignored'`` — the engine reads no source geometry, so the depths reach
#: no deck; ``'refused'`` — the default mode has no per-source sum.
_MULTI_DEPTH_IN_DEFAULT_MODE = {
    'Bellhop': 'field', 'Kraken': 'field', 'Scooter': 'field',
    'SPARC': 'field', 'Bounce': 'ignored', 'OAST': 'field',
    'OASN': 'refused', 'OASR': 'refused', 'OASP': 'field',
    'OASSP': 'field', 'OASS': 'refused', 'RAM': 'field',
}


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_a_multi_depth_source_stacks_in_a_field_mode_and_is_refused_elsewhere(
        model_name):
    """What the flag costs a caller, per model.

    Bellhop runs the grid. Bounce reads no source geometry and overrides
    ``_validate_geometry`` to a no-op, so it accepts the extra depths and
    they reach no deck. Every other model accepts a multi-depth Source in
    a field mode — ``run()`` loops over the depths — and refuses it in a
    mode with no per-source sum (OASN's array products, OASR's reflection
    table), naming the field modes that do stack.
    """
    try:
        m = build_engine(model_name)
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    source = uacpy.Source(depths=[30.0, 60.0], frequencies=100.0)
    args = (_reference_environment(model_name), source,
            _reference_receiver())
    expected = _MULTI_DEPTH_IN_DEFAULT_MODE[model_name]
    default_is_field = m._default_run_mode() in m._FIELD_MODES
    assert default_is_field == (expected == 'field')
    if expected in ('field', 'ignored'):
        m.validate_inputs(*args)
    else:
        with pytest.raises(ConfigurationError,
                           match='single source depth per .* run'):
            m.validate_inputs(*args)


_EXPECTED_ROUGH_BOTTOM = {
    'Bellhop': False, 'Kraken': True, 'Scooter': False, 'SPARC': False,
    'Bounce': False, 'OAST': True, 'OASN': True, 'OASR': True,
    'OASP': True, 'OASSP': True, 'OASS': True, 'RAM': False,
}


# Whether the engine honours ``env.absorption``. One flag, mirroring
# ``spec.traits.consumes_volume_absorption``: a second one would differ only on Bounce,
# and False is the honest answer there -- it tabulates R(theta) AT an
# interface, so there is no range over which volume loss accumulates.
_VOLUME_ATTENUATION = {
    'Bellhop': True,    # alpha_dB_per_m, carried in the imaginary travel time
    'Kraken': True,     # TopOpt position 4
    'Scooter': True,    # TopOpt position 4
    'SPARC': False,     # the march is lossless: sparc.f90:221 keeps Re(c) only
    'Bounce': False,    # a reflection table has no path length
    'RAM': True,        # water block on ksqw / lamw in every patched backend
    'OAST': True,       # water-layer AC in dB/wavelength (oases_writer._water_ac)
    'OASN': True,
    'OASP': True,       # one AC per layer, at the deck's centre frequency
    'OASR': False,      # a lossless water half-space, no path length
    'OASSP': True,      # OASP's deck writer (_write_oasp_family_deck)
    'OASS': True,       # water-layer AC, as OAST (write_oass_input)
}


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_volume_attenuation_capability_matrix(model_name):
    """Whether ``env.absorption`` reaches the engine, asked of the public API.

    RAM is the case this exists for: no backend puts loss in the water --
    ``matrc`` assigns ``ksq(i)=ksqw(i)`` above the bathymetry with no branch
    and ``ksqw`` carries no attenuation term -- so a caller has to be able to
    find that out without reading Fortran.
    """
    model = build_engine(model_name)
    assert model.supports_feature('volume_attenuation') is \
        _VOLUME_ATTENUATION[model_name], model_name


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_rough_bottom_capability_matrix(model_name):
    """Only solvers whose deck reaches a slot the binary reads may declare it.

    Kraken does: ``Kraken/kraken.f90:902`` feeds ``SSP%sigma(Medium+1)`` to
    ``KupIng``, which at ``Medium == LastAcoustic`` is the seabed interface.
    Scooter does not — the writer puts the half-space sigma on the BotOpt line
    (``write_bottom_section`` → ``SSP%sigma(NMedia+1)``) and no line in
    ``Scooter/`` reads that slot, while a *layer* sigma lands in the
    ``sigma(2:NMedia)`` range that ``Scooter/scooter.f90:63`` ERROUTs on. The
    OASES family reads it as column 7 (RG) of each layer record
    (``oases/src/oaseun31.f:54``).
    """
    try:
        m = build_engine(model_name)
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    assert m._supports_rough_bottom is _EXPECTED_ROUGH_BOTTOM[model_name]


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
