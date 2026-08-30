"""Supported-RunMode harmonization tests.

Locks in the per-model ``_supported_modes`` set — the mode-axis analogue of
``test_capability_flags``. If a model gains or loses a RunMode, the change must
come with an update here so the public mode surface stays explicit, and so
every *unsupported* mode keeps being refused.
"""

import pytest

from uacpy.models.base import RunMode
from uacpy.models.bellhop import Bellhop
from uacpy.models.kraken import Kraken
from uacpy.models.scooter import Scooter
from uacpy.models.sparc import SPARC
from uacpy.models.bounce import Bounce
from uacpy.models.oases import OAST, OASN, OASR, OASP, OASS, OASSP
from uacpy.models.ram import RAM
from uacpy.core.exceptions import ExecutableNotFoundError


# (model factory, expected supported RunModes). Lambdas because constructors
# resolve their binary eagerly.
_EXPECTED = {
    'Bellhop': (
        lambda: Bellhop(),
        {RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.SEMICOHERENT_TL,
         RunMode.RAYS, RunMode.EIGENRAYS, RunMode.ARRIVALS,
         RunMode.BROADBAND, RunMode.TIME_SERIES},
    ),
    'Kraken': (
        lambda: Kraken(),
        {RunMode.MODES, RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
         RunMode.BROADBAND, RunMode.TIME_SERIES},
    ),
    'Scooter': (
        lambda: Scooter(),
        {RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES},
    ),
    'SPARC': (
        lambda: SPARC(),
        # CW transmission loss withdrawn: the pulse-to-CW extraction is not
        # quantitative. SPARC's product is its native time series.
        {RunMode.TIME_SERIES},
    ),
    'Bounce': (
        lambda: Bounce(),
        {RunMode.REFLECTION},
    ),
    'OAST': (
        lambda: OAST(),
        {RunMode.COHERENT_TL},
    ),
    'OASN': (
        lambda: OASN(),
        {RunMode.COVARIANCE, RunMode.REPLICA},
    ),
    'OASR': (
        lambda: OASR(),
        {RunMode.REFLECTION},
    ),
    'OASP': (
        lambda: OASP(),
        {RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES},
    ),
    'OASS': (
        lambda: OASS(correlation_length=10.0),
        {RunMode.REVERBERATION, RunMode.COVARIANCE},
    ),
    'OASSP': (
        lambda: OASSP(correlation_length=10.0),
        {RunMode.BROADBAND, RunMode.TIME_SERIES},
    ),
    'RAM': (
        lambda: RAM(),
        {RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES},
    ),
}


_OASES_MODELS = {'OAST', 'OASN', 'OASR', 'OASP', 'OASS', 'OASSP'}


def _model_param(name):
    """Wrap parametrize values with the binary markers each model needs
    (every constructor existence-checks its binary), plus ``requires_oases``
    for the separately-licensed OASES family."""
    marks = [pytest.mark.requires_binary]
    if name in _OASES_MODELS:
        marks.append(pytest.mark.requires_oases)
    return pytest.param(name, marks=marks, id=name)


_MODEL_PARAMS = [_model_param(n) for n in _EXPECTED.keys()]


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_supported_modes(model_name):
    """``_supported_modes`` matches exactly; ``supports_mode`` agrees on the
    full RunMode enum (every unsupported mode refused)."""
    factory, expected = _EXPECTED[model_name]
    try:
        m = factory()
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")

    assert set(m._supported_modes) == expected, (
        f"{model_name}._supported_modes = {set(m._supported_modes)}, "
        f"expected {expected}"
    )
    for mode in RunMode:
        assert m.supports_mode(mode) is (mode in expected), (
            f"{model_name}.supports_mode({mode}) disagrees with "
            f"_supported_modes"
        )


# ── the reverse map: which models each ``compute_*`` names ─────────────────

#: ``PropagationModel.compute_*`` -> the ``RunMode`` it gates on. The
#: ``alternatives=[...]`` list each one raises with is the only guidance a
#: user gets when their model cannot answer, and it is written by hand
#: because the mapping mode -> models cannot be derived inside ``base.py``
#: (it would need every wrapper imported, and ``base.py`` is what they
#: import). That makes it a hand-maintained copy of ``spec.modes``, which is
#: what the test below compares it against.
_COMPUTE_METHOD_MODES = {
    'compute_tl': RunMode.COHERENT_TL,
    'compute_rays': RunMode.RAYS,
    'compute_arrivals': RunMode.ARRIVALS,
    'compute_modes': RunMode.MODES,
    'compute_eigenrays': RunMode.EIGENRAYS,
    'compute_reflection': RunMode.REFLECTION,
    'compute_time_series': RunMode.TIME_SERIES,
    'compute_transfer_function': RunMode.BROADBAND,
    'compute_covariance': RunMode.COVARIANCE,
    'compute_replicas': RunMode.REPLICA,
}


def _hardcoded_alternatives(method_name):
    """The ``alternatives=[...]`` literal in ``base.py``'s ``method_name``."""
    import ast
    import inspect
    from uacpy.models import base as base_mod

    tree = ast.parse(inspect.getsource(base_mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            for call in ast.walk(node):
                if (isinstance(call, ast.Call)
                        and getattr(call.func, 'id', '')
                        == 'UnsupportedFeatureError'):
                    for kw in call.keywords:
                        if kw.arg == 'alternatives':
                            return [e.value for e in kw.value.elts]
    raise AssertionError(f"no alternatives=[...] found in {method_name}")


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
    """The hand-written advice equals the declared truth.

    Without this, a model that gains a mode never appears in the message that
    sends users to it, and one that loses a mode keeps being recommended —
    both silent, because the list is a string literal nothing reads back.
    """
    mode = _COMPUTE_METHOD_MODES[method_name]
    assert sorted(_hardcoded_alternatives(method_name)) == _models_declaring(mode)


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
    assert set(found) == {
        'Bellhop', 'Bounce', 'Kraken', 'OASN', 'OASP', 'OASR', 'OASS',
        'OASSP', 'OAST', 'RAM', 'SPARC', 'Scooter'}


# ── what a concrete subclass has to declare ───────────────────────────────


def _concrete_double(**namespace):
    """Build a ``PropagationModel`` subclass with ``run`` and ``namespace``."""
    from uacpy.models.base import PropagationModel

    body = {'run': lambda self, env, source, receiver, run_mode=None: None}
    body.update(namespace)
    return type('Double', (PropagationModel,), body)


class TestAConcreteWrapperMustDeclareSpecAndSource:
    """A subclass that defines ``run()`` is one a user can hold, so both
    declarations are required at class-definition time.

    Without ``spec`` the class silently takes the base defaults — COHERENT_TL
    only, no env-shape support, point sources — which is nobody's real
    answer. Without ``source`` the licence and citation path is skipped
    outright: ``_warn_restricted_source`` returns immediately on
    ``source is None``, so a restricted engine would be wrapped with no
    warning.
    """

    def test_a_subclass_with_both_is_accepted(self):
        from uacpy.models.base import ModelSpec
        cls = _concrete_double(spec=ModelSpec(modes=(RunMode.COHERENT_TL,)),
                               source='acoustics_toolbox')
        assert cls.source == 'acoustics_toolbox'

    def test_neither_is_refused_and_both_are_named(self):
        with pytest.raises(TypeError, match='declares no spec or source'):
            _concrete_double()

    def test_a_missing_source_alone_is_refused(self):
        """The licence leg on its own — the half with real weight."""
        from uacpy.models.base import ModelSpec
        with pytest.raises(TypeError, match='declares no source'):
            _concrete_double(spec=ModelSpec(modes=(RunMode.COHERENT_TL,)))

    def test_a_missing_spec_alone_is_refused(self):
        with pytest.raises(TypeError, match='declares no spec'):
            _concrete_double(source='acoustics_toolbox')

    def test_an_intermediate_base_that_defines_no_run_is_left_alone(self):
        """``OASES`` declares neither and must stay legal: it leaves ``run``
        abstract, so the two declarations are its subclasses' to make."""
        from uacpy.models.base import PropagationModel
        from uacpy.models.oases import OASES
        assert 'spec' not in OASES.__dict__
        assert 'source' not in OASES.__dict__
        assert OASES.run is PropagationModel.run
        type('Intermediate', (PropagationModel,), {})

    @pytest.mark.parametrize('model_name', _MODEL_PARAMS)
    def test_every_shipped_wrapper_already_declares_both(self, model_name):
        factory = _EXPECTED[model_name][0]
        try:
            model = factory()
        except ExecutableNotFoundError:
            pytest.skip(f"{model_name} binary not available")
        assert model.spec is not None
        assert model.source is not None
