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
from uacpy.models.kraken import Kraken
from uacpy.models.scooter import Scooter
from uacpy.models.sparc import SPARC
from uacpy.models.bounce import Bounce
from uacpy.models.oases import OAST, OASN, OASR, OASP
from uacpy.models.ram import RAM
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, UnsupportedFeatureError,
)
from uacpy.models.base import VALID_SOURCE_TYPES


_FEATURES = (
    'altimetry',
    'range_dependent_bathymetry',
    'range_dependent_ssp',
    'range_dependent_bottom',
    'layered_bottom',
    'elastic_media',
)


# (model factory, expected flags by feature). Use lambdas because every
# constructor resolves its binary eagerly, so construction must stay behind
# the markers ``_model_param`` attaches.
_EXPECTED = {
    'Bellhop': (
        lambda: Bellhop(),
        {'altimetry': True, 'range_dependent_bathymetry': True,
         'range_dependent_ssp': True,
         'range_dependent_bottom': True, 'layered_bottom': False,
         'elastic_media': True},
    ),
    'Kraken': (
        lambda: Kraken(),
        {'altimetry': False, 'range_dependent_bathymetry': True,
         'range_dependent_ssp': True,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    ),
    'Scooter': (
        lambda: Scooter(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    ),
    'SPARC': (
        lambda: SPARC(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': False},
    ),
    'Bounce': (
        lambda: Bounce(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    ),
    'OAST': (
        lambda: OAST(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    ),
    'OASN': (
        lambda: OASN(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    ),
    'OASR': (
        lambda: OASR(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    ),
    'OASP': (
        lambda: OASP(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'elastic_media': True},
    ),
    'RAM': (
        lambda: RAM(),
        {'altimetry': True, 'range_dependent_bathymetry': True,
         'range_dependent_ssp': True,
         'range_dependent_bottom': True, 'layered_bottom': True,
         'elastic_media': True},
    ),
}


_OASES_MODELS = {'OAST', 'OASN', 'OASR', 'OASP'}


def _model_param(name):
    """Wrap parametrize values with the binary markers each model needs.

    Every model resolves (and existence-checks) its binary in ``__init__``,
    so this test — which constructs the model to read its capability flags —
    needs ``requires_binary`` for all, plus ``requires_oases`` for the
    separately-licensed OASES family. Lets ``pytest -m 'not requires_binary'``
    / ``'not requires_oases'`` deselect at collection time.
    """
    marks = [pytest.mark.requires_binary]
    if name in _OASES_MODELS:
        marks.append(pytest.mark.requires_oases)
    return pytest.param(name, marks=marks, id=name)


_MODEL_PARAMS = [_model_param(n) for n in _EXPECTED.keys()]


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
@pytest.mark.parametrize('feature', _FEATURES)
def test_capability_flag(model_name, feature):
    factory, expected = _EXPECTED[model_name]
    try:
        m = factory()
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
    'OAST':    ({'point'}, False),
    'OASN':    ({'point'}, False),
    # The one OASES class that widens past 'point': a plane-wave reflection
    # coefficient does not depend on source geometry, and OASR's deck writer
    # reads only source.frequencies — same reasoning as Bounce, same run mode.
    'OASR':    ({'point', 'line', 'scaled'}, False),
    'OASP':    ({'point'}, False),
    'RAM':     ({'point'}, False),
}


def _reference_environment():
    return uacpy.Environment(
        bathymetry=200.0,
        ssp=uacpy.SoundSpeedProfile(depths=[0, 200], data=[1500, 1500]),
    )


def _reference_receiver():
    return uacpy.Receiver(depths=100.0, ranges=np.linspace(100, 2000, 20))


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_source_capability_matrix(model_name):
    """Locks the per-model source-geometry / beam-pattern surface."""
    factory = _EXPECTED[model_name][0]
    try:
        m = factory()
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
    assert np.any(np.asarray(reference.R) != 0.0), (
        "the reference coefficient is identically zero — this fixture cannot "
        "tell an unchanged answer from an absent one")
    assert np.array_equal(np.asarray(result.R), np.asarray(reference.R))


def test_every_declared_source_type_is_valid():
    for types, _ in _EXPECTED_SOURCE_TYPES.values():
        assert types <= VALID_SOURCE_TYPES


@pytest.mark.requires_binary
def test_sparc_honours_no_source_geometry():
    # A source geometry is a weighting inside the wavenumber->range Hankel
    # transform, and only the snapshot mode runs one (``SPARC._run_snapshot``
    # hands ``source_type`` to ``sparc_snapshot_to_time_field``). The default
    # ``output_mode='R'`` and ``'D'`` are range- / depth-native and never
    # reach it, so they honour no geometry beyond a point source.
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
            m = _EXPECTED[model_name][0]()
        except ExecutableNotFoundError:
            pytest.skip(f"{model_name} binary not available")
        from uacpy.models.base import _CAPABILITY_FLAGS
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
        with pytest.raises(ValueError, match='unknown capability'):
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
            clinear = Bellhop(interp_ssp='c-linear')
        except ExecutableNotFoundError:
            pytest.skip("Bellhop binary not available")
        assert 'range_dependent_ssp' not in Bellhop.spec.supports
        assert quad.supports_feature('range_dependent_ssp') is True
        assert clinear.supports_feature('range_dependent_ssp') is False


_EXPECTED_ROUGH_SURFACE = {
    'Bellhop': False, 'Kraken': True, 'Scooter': True, 'SPARC': False,
    'Bounce': False, 'OAST': True, 'OASN': True,
    # OASR's deck has no sea surface: its layer 1 is the water half-space
    # the plane wave arrives through, whose RG INENVI discards
    # (oaseun31.f:377), so surface roughness is collapsed with a warning.
    'OASR': False,
    'OASP': True, 'RAM': False,
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
        m = _EXPECTED[model_name][0]()
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
        m = _EXPECTED['SPARC'][0]()
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
    'Bellhop': True, 'Kraken': False, 'Scooter': False, 'SPARC': False,
    'Bounce': False, 'OAST': False, 'OASN': False, 'OASR': False,
    'OASP': False, 'RAM': False,
}


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_multi_source_depth_capability_matrix(model_name):
    """Bellhop is the only model that runs a source-depth *grid* in one
    binary call (``ModelSpec.supports`` carries ``'multi_source_depth'``);
    for every other model ``_validate_geometry`` refuses a multi-depth
    Source with 'single source depth' (pinned in test_input_validation.py),
    telling the caller to loop over Sources externally. Bounce also reads
    ``False``, but its geometry validation is a no-op so nothing enforces it.
    """
    try:
        m = _EXPECTED[model_name][0]()
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    assert (m._supports_multi_source_depth
            is _EXPECTED_MULTI_SOURCE_DEPTH[model_name])


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_a_multi_depth_source_raises_or_reaches_no_deck(model_name):
    """What the flag costs a caller, per model.

    Bellhop runs the grid. Bounce reads no source geometry and overrides
    ``_validate_geometry`` to a no-op, so it accepts the extra depths and
    they reach no deck. Every other model raises and names the loop the
    caller has to write — nothing loops over source depths inside uacpy.
    """
    try:
        m = _EXPECTED[model_name][0]()
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    source = uacpy.Source(depths=[30.0, 60.0], frequencies=100.0)
    args = (_reference_environment(), source, _reference_receiver())
    if model_name in ('Bellhop', 'Bounce'):
        m.validate_inputs(*args)
    else:
        with pytest.raises(ConfigurationError,
                           match='single source depth per run'):
            m.validate_inputs(*args)


_EXPECTED_ROUGH_BOTTOM = {
    'Bellhop': False, 'Kraken': True, 'Scooter': False, 'SPARC': False,
    'Bounce': False, 'OAST': True, 'OASN': True, 'OASR': True,
    'OASP': True, 'RAM': False,
}


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
        m = _EXPECTED[model_name][0]()
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    assert m._supports_rough_bottom is _EXPECTED_ROUGH_BOTTOM[model_name]


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
        assert projected.surface.properties[0].roughness == 0.0, (
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
        assert np.all(np.isfinite(result.db))


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
        assert np.nanmax(np.abs(smooth.db - rough.db)) > 1.0

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
