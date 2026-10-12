"""Run-mode surface tests.

Which run modes each engine declares and resolves is held to its saved record
(``tests/data/engine_specs.json``, ``test_capability_flags.py``). The tests
here hold what is read off those modes: ``supports_mode`` answers for every
``RunMode`` member as ``supported_modes`` does, the ``compute_*`` refusals name
exactly the engines that run their mode, every constructor knob is
documented, and a concrete wrapper declares a well-formed spec.
"""

import pytest

from uacpy.core.run_settings import RunMode
from uacpy.core.exceptions import ExecutableNotFoundError
from uacpy.tests.conftest import build_engine, engine_names, engine_params


_MODEL_PARAMS = engine_params(value='name')


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_supports_mode_answers_as_supported_modes(model_name):
    """``supports_mode`` agrees with ``supported_modes`` on the full RunMode
    enum: every supported mode accepted, every other one refused."""
    try:
        m = build_engine(model_name)
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    for mode in RunMode:
        assert m.supports_mode(mode) is (mode in m.supported_modes), (
            f"{model_name}.supports_mode({mode}) disagrees with "
            f"supported_modes"
        )


# ── the reverse map: which models each ``compute_*`` names ─────────────────

#: ``PropagationModel.compute_*`` -> the ``RunMode`` it gates on. The
#: ``alternatives`` list each one raises with is the only guidance a user
#: gets when their model cannot answer; it is read from the engine registry
#: (``uacpy.models._registry.engines_running``) when the refusal is raised,
#: and the tests below compare it with every wrapper's ``spec.modes``.
_COMPUTE_METHOD_MODES = {
    'compute_field': RunMode.COHERENT_TL,
    'compute_rays': RunMode.RAYS,
    'compute_arrivals': RunMode.ARRIVALS,
    'compute_modes': RunMode.MODES,
    'compute_eigenrays': RunMode.EIGENRAYS,
    'compute_reflection': RunMode.REFLECTION,
    'compute_time_series': RunMode.TIME_SERIES,
    'compute_transfer_function': RunMode.BROADBAND,
    'compute_covariance': RunMode.COVARIANCE,
    'compute_replicas': RunMode.REPLICA,
    'compute_reverberation': RunMode.REVERBERATION,
}


def _refusal_alternatives(method_name):
    """The ``alternatives`` a model lacking the method's mode is refused
    with: a stand-in model that runs only some other mode calls it."""
    import uacpy
    from uacpy.models.base import PropagationModel
    from uacpy.models._spec import ModelSpec

    mode = _COMPUTE_METHOD_MODES[method_name]
    other = (RunMode.REVERBERATION if mode != RunMode.REVERBERATION
             else RunMode.COHERENT_TL)

    def _never_launched(self, *args):
        raise AssertionError('the refusal comes before any run')

    class _OtherModeOnly(PropagationModel):
        spec = ModelSpec(modes=(other,))
        provenance_id = 'acoustics_toolbox'
        _write_input = _launch = _read_output = _to_result = _never_launched

    env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=[20.0], ranges=[1000.0])
    method = getattr(_OtherModeOnly(), method_name)
    args = (env, src) if method_name == 'compute_modes' else (env, src, rcv)
    with pytest.raises(uacpy.UnsupportedFeatureError,
                       match='_OtherModeOnly does not support: ') as info:
        method(*args)
    return info.value.alternatives


def _models_declaring(mode):
    """Every concrete wrapper whose ``spec.modes`` carries ``mode``."""
    import inspect
    from uacpy.models.base import PropagationModel
    import uacpy.models as models_pkg

    out = []
    for name in dir(models_pkg):
        obj = getattr(models_pkg, name)
        if (inspect.isclass(obj) and issubclass(obj, PropagationModel)
                and obj is not PropagationModel
                and not inspect.isabstract(obj)
                and getattr(obj, 'spec', None) is not None
                and mode in obj.spec.modes):
            out.append(obj.__name__)
    return sorted(set(out))


@pytest.mark.parametrize('method_name', sorted(_COMPUTE_METHOD_MODES))
def test_compute_method_names_every_model_declaring_its_mode(method_name):
    """The advice a refusal gives equals the declared truth: every wrapper
    whose ``spec.modes`` carries the mode, sorted, and nothing else. A model
    that gains a mode appears in the message that sends users to it, and one
    that loses a mode stops being recommended."""
    mode = _COMPUTE_METHOD_MODES[method_name]
    assert _refusal_alternatives(method_name) == _models_declaring(mode)
    assert _models_declaring(mode), f"no engine runs {mode.name}"


def test_every_compute_method_is_covered_by_the_reverse_map():
    """A new ``compute_*`` with its own ``alternatives=[...]`` has to be added
    to ``_COMPUTE_METHOD_MODES``, or the gate above silently skips it."""
    import inspect
    from uacpy.models.base import PropagationModel

    found = {
        name for name, _ in inspect.getmembers(PropagationModel,
                                               inspect.isfunction)
        if name.startswith('compute_')
    }
    assert found == set(_COMPUTE_METHOD_MODES)


def test_every_model_constructor_parameter_is_documented():
    """No constructor knob is reachable but unwritten.

    Read across the class docstring *and* the ``__init__`` docstring together,
    because the twelve wrappers split the Parameters section between them
    differently and either placement is a real answer for the user. The gap
    this closes is a whole family of parameters going undescribed at once —
    the six plumbing arguments every model forwards to
    ``PropagationModel.__init__`` are documented as one combined entry, and a
    model that omits the entry omits all six.
    """
    import inspect
    import re

    import uacpy.models as models_pkg
    from uacpy.models.base import PropagationModel

    undocumented = {}
    for name in dir(models_pkg):
        cls = getattr(models_pkg, name)
        if not (inspect.isclass(cls) and issubclass(cls, PropagationModel)
                and cls is not PropagationModel
                and getattr(cls, 'spec', None) is not None):
            continue
        doc = f"{cls.__doc__ or ''}\n{cls.__init__.__doc__ or ''}"
        missing = [
            param for param in inspect.signature(cls.__init__).parameters
            if param != 'self'
            and not re.search(rf'(?<![\w]){re.escape(param)}(?![\w])', doc)
        ]
        if missing:
            undocumented[cls.__name__] = missing
    assert not undocumented, (
        f"constructor parameter(s) with no docstring entry: {undocumented}")


def test_the_constructor_documentation_sweep_reads_every_wrapper():
    """A sweep that collected no classes would pass the gate above."""
    import inspect

    import uacpy.models as models_pkg
    from uacpy.models.base import PropagationModel

    found = [
        getattr(models_pkg, name).__name__ for name in dir(models_pkg)
        if inspect.isclass(getattr(models_pkg, name))
        and issubclass(getattr(models_pkg, name), PropagationModel)
        and getattr(models_pkg, name) is not PropagationModel
        and getattr(getattr(models_pkg, name), 'spec', None) is not None
    ]
    assert set(found) == engine_names()


# ── what a concrete subclass has to declare ───────────────────────────────


def _concrete_double(**namespace):
    """Build a ``PropagationModel`` subclass with ``run`` and ``namespace``."""
    from uacpy.models.base import PropagationModel

    body = {'run': lambda self, env, source, receiver, run_mode=None: None}
    body.update(namespace)
    return type('Double', (PropagationModel,), body)


class TestAConcreteWrapperMustDeclareSpecAndProvenanceId:
    """A subclass that defines ``run()`` is one a user can hold, so both
    declarations are required at class-definition time.

    Without ``spec`` the class silently takes the base defaults — COHERENT_TL
    only, no env-shape support, point sources — which is nobody's real
    answer. Without ``provenance_id`` the licence and citation path is skipped
    outright: ``_warn_restricted_provenance`` returns immediately on
    ``provenance_id is None``, so a restricted engine would be wrapped with no
    warning.
    """

    def test_a_subclass_with_both_is_accepted(self):
        from uacpy.models._spec import ModelSpec
        cls = _concrete_double(spec=ModelSpec(modes=(RunMode.COHERENT_TL,)),
                               provenance_id='acoustics_toolbox')
        assert cls.provenance_id == 'acoustics_toolbox'

    def test_neither_is_refused_and_both_are_named(self):
        with pytest.raises(TypeError, match='declares no spec or provenance_id'):
            _concrete_double()

    def test_a_missing_provenance_id_alone_is_refused(self):
        """The licence leg on its own — the half with real weight."""
        from uacpy.models._spec import ModelSpec
        with pytest.raises(TypeError, match='declares no provenance_id'):
            _concrete_double(spec=ModelSpec(modes=(RunMode.COHERENT_TL,)))

    def test_a_missing_spec_alone_is_refused(self):
        with pytest.raises(TypeError, match='declares no spec'):
            _concrete_double(provenance_id='acoustics_toolbox')

    def test_an_intermediate_base_that_defines_no_run_is_left_alone(self):
        """``OASES`` declares neither and must stay legal: it leaves ``run``
        abstract, so the two declarations are its subclasses' to make."""
        from uacpy.models.base import PropagationModel
        from uacpy.models.oases import OASES
        assert 'spec' not in OASES.__dict__
        assert 'provenance_id' not in OASES.__dict__
        assert OASES.run is PropagationModel.run
        type('Intermediate', (PropagationModel,), {})

    @pytest.mark.parametrize('model_name', _MODEL_PARAMS)
    def test_every_shipped_wrapper_already_declares_both(self, model_name):
        try:
            model = build_engine(model_name)
        except ExecutableNotFoundError:
            pytest.skip(f"{model_name} binary not available")
        assert model.spec is not None
        assert model.provenance_id is not None


class TestTheTraitsAreCheckedWhenTheClassIsDefined:
    """``spec.traits`` is read on every call (the stacking rule, the band
    notices, the keyword rule), so a malformed one fails on import, where
    the spec itself is checked, rather than deep inside a run."""

    @staticmethod
    def _spec(**traits):
        from uacpy.models._spec import EngineTraits, ModelSpec
        return ModelSpec(modes=(RunMode.COHERENT_TL,),
                         traits=EngineTraits(**traits))

    def test_a_mode_set_of_run_modes_is_accepted(self):
        cls = _concrete_double(
            spec=self._spec(native_multi_depth_modes=frozenset(
                {RunMode.COHERENT_TL})),
            provenance_id='acoustics_toolbox')
        assert cls.spec.traits.native_multi_depth_modes == {
            RunMode.COHERENT_TL}

    def test_a_mode_set_holding_a_string_is_refused(self):
        with pytest.raises(TypeError,
                           match='traits.native_multi_depth_modes must be a '
                                 'frozenset of RunMode'):
            _concrete_double(
                spec=self._spec(native_multi_depth_modes=frozenset(
                    {'coherent_tl'})),
                provenance_id='acoustics_toolbox')

    def test_a_mode_set_that_is_not_a_frozenset_is_refused(self):
        with pytest.raises(TypeError,
                           match='traits.announced_band_modes must be a '
                                 'frozenset of RunMode'):
            _concrete_double(
                spec=self._spec(announced_band_modes={RunMode.BROADBAND}),
                provenance_id='acoustics_toolbox')

    def test_the_benign_fatals_are_a_tuple_of_strings(self):
        _concrete_double(spec=self._spec(benign_fortran_fatals=('No modes',)),
                         provenance_id='acoustics_toolbox')
        with pytest.raises(TypeError, match='benign_fortran_fatals must be '
                                            'a tuple of strings'):
            _concrete_double(
                spec=self._spec(benign_fortran_fatals=['No modes']),
                provenance_id='acoustics_toolbox')

    def test_traits_of_another_type_are_refused(self):
        from uacpy.models._spec import ModelSpec
        with pytest.raises(TypeError, match='spec.traits must be an '
                                            'EngineTraits, got dict'):
            _concrete_double(
                spec=ModelSpec(modes=(RunMode.COHERENT_TL,),
                               traits={'consumes_run_t_start': True}),
                provenance_id='acoustics_toolbox')
