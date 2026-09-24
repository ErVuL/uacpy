"""Rules about the SHAPE of the package, not the value of any number.

Every rule here was a defect first. Each one was found by reading, once, and
fixed by hand; these exist so the next instance is found by the suite instead.
They are deliberately narrow: a rule that cannot say precisely what is wrong
is a rule that gets suppressed.

* :func:`test_no_plotter_branches_on_a_model_name` — a plotter draws what it
  is handed. ``plot_absorption(frequencies, model='thorp')`` both selected a
  formula and drew it, which made it a second spelling of
  ``absorption_thorp(f).plot()``; the name-keyed map inside it had no other
  caller in the package, so the map had no reason to exist either.
* :func:`test_no_vocabulary_is_declared_twice` — ``LOSS_KINDS`` was declared
  in the data layer and again in the plotting layer, beside a comment
  asserting it was declared once. A third kind added to one would have made
  ``Field.max`` report the loud end while the 1-D cut drew its axis the other
  way.
* :func:`test_a_decibel_is_never_spelled_db` — ``db`` is a database and ``dB``
  is a decibel. Two identifiers broke it (``vmin_db``, ``diag_db``); the ten
  that keep it are real databases and are listed.
"""
import ast
import collections
from pathlib import Path

import pytest

import uacpy

_PACKAGE = Path(uacpy.__file__).resolve().parent
_SKIP = ('__pycache__', 'third_party', '/build/', '/tests/', '/examples/')


def _sources():
    """Every first-party module, excluding vendored and generated trees."""
    for path in sorted(_PACKAGE.rglob('*.py')):
        if any(part in str(path) for part in _SKIP):
            continue
        try:
            yield path, ast.parse(path.read_text(encoding='utf-8'))
        except (SyntaxError, UnicodeDecodeError):       # pragma: no cover
            continue


def _rel(path):
    return str(path.relative_to(_PACKAGE.parent))


#: Identifiers that end in ``_db`` and mean a *database*. Anything else ending
#: that way is a decibel wearing the wrong case and fails the rule.
_KNOWN_DATABASES = frozenset({
    'download_crust1_db', 'download_diesing_db', 'download_emodnet_db',
    'download_globsed_db', 'download_glodap_db', 'download_graw_db',
    'download_seaice_db', 'download_sediment_db', 'download_wind_db',
    'sediment_db',
})

#: Parameters whose value names a formula, model or backend. A plotter that
#: branches on one of these is choosing physics, not drawing it.
_SELECTOR_NAMES = frozenset({'model', 'formula', 'method', 'backend'})


def _selects_by_name(func_node, selectors):
    """Where ``func_node`` uses a selector parameter to CHOOSE something.

    Three spellings, because the rule originally saw only the first and so
    could not catch the construct its own docstring cites — a registry
    subscript is how ``plot_absorption`` would most naturally have been
    written, and reinstating it passed the suite:

    * ``model == 'thorp'`` / ``model in ('fg', …)``  — a comparison
    * ``_FORMULAS[model]``                            — a registry subscript
    * ``_FORMULAS.get(model)``                        — the same, softened

    ``model.lower()`` and ``str(model)`` wrap the name before any of these, so
    the operand is unwrapped first.
    """
    def names_a_selector(node):
        while isinstance(node, ast.Call) and node.args:
            node = node.args[0] if not isinstance(node.func, ast.Attribute) \
                else node.func.value                    # model.lower() / str(model)
        if isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name) and node.id.lstrip('_') in selectors

    for inner in ast.walk(func_node):
        if isinstance(inner, ast.Compare):
            if names_a_selector(inner.left):
                yield inner.lineno, 'compares'
        elif isinstance(inner, ast.Subscript):
            if names_a_selector(inner.slice):
                yield inner.lineno, 'indexes a registry with'
        elif (isinstance(inner, ast.Call)
              and isinstance(inner.func, ast.Attribute)
              and inner.func.attr in ('get', 'pop')
              and inner.args and names_a_selector(inner.args[0])):
            yield inner.lineno, 'looks up'
        elif isinstance(inner, ast.Match) and names_a_selector(inner.subject):
            yield inner.lineno, 'matches on'


def test_no_plotter_branches_on_a_model_name():
    """A ``plot_*`` function, or any method named ``plot``, may not select a
    formula by name.

    Drawing and choosing are separate jobs. When they are the same call the
    computation has no entry point of its own — the only way to get the
    numbers is to draw them — and the plotter grows a private registry nothing
    else can reach. That is what ``plot_absorption`` was.
    """
    offenders = []
    for path, tree in _sources():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Carriers spell it ``plot``; module-level plotters ``plot_*``.
            if not (node.name.startswith('plot_') or node.name == 'plot'):
                continue
            args = {a.arg for a in node.args.args + node.args.kwonlyargs}
            selectors = {a.lstrip('_') for a in args} & _SELECTOR_NAMES
            if not selectors:
                continue
            # Follow local aliases to a fixed point. The original code read
            # ``m = str(model).lower().replace('-', '_')`` and then compared
            # ``m``, so a rule watching only the parameter itself missed the
            # very construct it was written for.
            selectors = set(selectors)
            for _ in range(4):
                grown = set(selectors)
                for assign in ast.walk(node):
                    if not isinstance(assign, ast.Assign):
                        continue
                    mentioned = {n.id.lstrip('_') for n in ast.walk(assign.value)
                                 if isinstance(n, ast.Name)}
                    if mentioned & selectors:
                        grown |= {t.id for t in assign.targets
                                  if isinstance(t, ast.Name)}
                if grown == selectors:
                    break
                selectors = grown
            for lineno, how in _selects_by_name(node, selectors):
                offenders.append(f'{_rel(path)}:{lineno}: {node.name} '
                                 f'{how} a model name')
    assert not offenders, (
        'a plotter is choosing physics by name instead of drawing what it '
        'was handed:\n  ' + '\n  '.join(sorted(set(offenders))))


def _string_vocabulary(node):
    """The string vocabulary a literal collection declares, or ``None``.

    Keyed on the **members**, and for a mapping on its **keys**: a registry
    maps a vocabulary to callables, so requiring constant values made the rule
    blind to ``_FORMULAS`` — the very duplication its docstring cites. Lists
    and ``frozenset({...})`` count too; each hid a live duplication when this
    rule looked only at tuples, sets and constant-valued dicts.
    """
    if isinstance(node, ast.Call):                      # frozenset({...}), set([...])
        func = node.func
        name = getattr(func, 'id', getattr(func, 'attr', None))
        if name in ('frozenset', 'set', 'tuple', 'list') and len(node.args) == 1:
            return _string_vocabulary(node.args[0])
        return None
    if isinstance(node, (ast.Tuple, ast.Set, ast.List)):
        members = tuple(sorted(e.value for e in node.elts
                               if isinstance(e, ast.Constant)
                               and isinstance(e.value, str)))
        return members if len(members) == len(node.elts) else None
    if isinstance(node, ast.Dict):
        # A mapping is compared on its keys AND values, so only a fully
        # identical table matches another. A dict keyed by a vocabulary is a
        # payload — ``RAY_CLASS_COLOURS`` gives each bounce class a colour —
        # and is not a second copy of that vocabulary. Those are held to
        # their owner by ``TestEveryKeyedTableCoversItsVocabulary`` instead.
        pairs = tuple(sorted(
            (k.value, repr(getattr(v, 'value', ast.dump(v))))
            for k, v in zip(node.keys, node.values)
            if isinstance(k, ast.Constant) and isinstance(k.value, str)))
        return ('dict',) + pairs if len(pairs) == len(node.keys) else None
    return None


def test_no_vocabulary_is_declared_twice():
    """The same set of string members may not be declared in two modules.

    One of them is a copy, and copies drift. This covers tuples, sets, lists,
    dict keys and ``frozenset({...})``, and both plain and annotated
    assignment, because every one of those spellings has hidden a real
    duplication in this package.
    """
    seen = collections.defaultdict(set)
    for path, tree in _sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            # ``__all__`` is a re-export list: naming the same symbols as the
            # module it re-exports from is the point, not a copy.
            if any(getattr(t, 'id', None) == '__all__' for t in targets):
                continue
            members = _string_vocabulary(value)
            if members and len(members) >= 2:
                seen[members].add(_rel(path))
    duplicated = {k: v for k, v in seen.items() if len(v) > 1}
    assert not duplicated, (
        'the same vocabulary is declared in more than one module; import it '
        'from the one that owns it:\n  ' + '\n  '.join(
            f'{list(k)} in {sorted(v)}' for k, v in sorted(duplicated.items())))


def test_a_decibel_is_never_spelled_db():
    """``db`` is a database; ``dB`` is a decibel. No exception for the kind of
    identifier — a parameter, a variable and an attribute all read the same
    way to whoever is scanning the line."""
    offenders = []
    for path, tree in _sources():
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.arg):
                names.append((node.arg, node.lineno))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.append((node.id, node.lineno))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Every entry in _KNOWN_DATABASES is a FUNCTION name, so the
                # whitelist is only reachable when function names are
                # inspected here.
                names.append((node.name, node.lineno))
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                # self.vmin_db — an attribute, which the docstring promises to
                # cover ("a parameter, a variable and an attribute").
                names.append((node.attr, node.lineno))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.append((node.target.id, node.target.lineno))
            elif isinstance(node, ast.Call):
                # g(vmin_db=1) — the name is written at the call, so a reader
                # scanning the line sees it whether or not the callee spells
                # it correctly.
                names.extend((kw.arg, node.lineno) for kw in node.keywords
                             if kw.arg)
            for name, lineno in names:
                if name.endswith('_db') and name not in _KNOWN_DATABASES:
                    offenders.append(f'{_rel(path)}:{lineno}: {name}')
    assert not offenders, (
        'a decibel spelled _db (or a database missing from the list in this '
        'file):\n  ' + '\n  '.join(sorted(set(offenders))))


class TestKnobsNothingElseEverPasses:
    """Public parameters with no caller anywhere in the repository.

    An unused knob is not a dead one — several here are whole alternative
    input modes, and removing them on a usage count would be the mistake.
    But unused also means *unverified*, and this package has already shipped
    one that could never work: `plot_mode_excitation(env=…)` called a method
    no version of uacpy has ever had, and a bare `except` turned that into a
    silent 1500 m/s.

    So these are executed rather than counted. Every one below was reported
    by an AST sweep as never passed anywhere, including tests and examples.
    """

    @staticmethod
    def _modes():
        import numpy as _np
        from uacpy.core.results.modes import Modes
        z = _np.linspace(0.0, 100.0, 51)
        k = 2 * _np.pi * 100.0 / _np.array([1600.0, 1520.0, 1450.0])
        phi = _np.stack([_np.sin((m + 1) * _np.pi * z / 100.0)
                         for m in range(3)], axis=1)
        return Modes(k=k, phi=phi, depths=z, frequencies=100.0, model='T')

    def test_modal_propagation_loss_source_density_changes_the_answer(self):
        """A knob that did nothing would be worse than an absent one."""
        import numpy as _np
        kw = dict(source_depth=50.0, receiver_depths=_np.array([50.0]),
                  ranges_m=_np.array([1000.0]))
        modes = self._modes()
        light = float(_np.asarray(
            modes.modal_propagation_loss(**kw, source_density=1.0).tl)[0, 0])
        heavy = float(_np.asarray(
            modes.modal_propagation_loss(**kw, source_density=2.0).tl)[0, 0])
        assert heavy > light + 1.0, (light, heavy)

    def test_the_alternative_and_cosmetic_knobs_all_execute(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as _np
        import uacpy
        from uacpy import acoustic_signal

        rng = _np.random.default_rng(0)
        t = _np.linspace(0.0, 1e-3, 64)
        x = _np.sin(2 * _np.pi * 5e3 * t)
        modes, src = self._modes(), uacpy.Source(depths=50.0,
                                                 frequencies=100.0)
        calls = {
            # a whole alternative input mode, not a toggle
            'plot_roc(pfa=, pd=)': lambda: uacpy.plot.plot_roc(
                pfa=_np.linspace(1e-3, 1.0, 20), pd=_np.linspace(0.0, 1.0, 20)),
            'cwt(w0=)': lambda: acoustic_signal.cwt(x, 1e6, w0=8.0),
            'wigner_ville(analytic=)': lambda: acoustic_signal.wigner_ville(
                x, 1e6, analytic=False),
            'plot_mode_excitation(floor_dB=)': lambda: uacpy.plot_mode_excitation(
                modes, src, sound_speed=1500.0, floor_dB=-30.0),
            'plot_mode_excitation(show_array_factor=)':
                lambda: uacpy.plot_mode_excitation(
                    modes, src, sound_speed=1500.0, show_array_factor=False),
            'plot_matched_field(mark_peak=)': lambda: uacpy.plot.plot_matched_field(
                _np.linspace(0.0, 5e3, 20), _np.linspace(0.0, 100.0, 15),
                _np.abs(rng.normal(size=(15, 20))) + 1e-6, mark_peak=False),
        }
        for label, call in calls.items():
            try:
                call()
            except Exception as exc:                    # pragma: no cover
                pytest.fail(f'{label} raised {type(exc).__name__}: {exc}')
            finally:
                plt.close('all')


class TestEveryKeyedTableCoversItsVocabulary:
    """A table keyed by a vocabulary must key on all of it, and only it.

    These are not duplicates — `RAY_CLASS_COLOURS` gives each bounce class a
    colour, which is a payload the vocabulary does not carry — so the
    duplicate-vocabulary rule leaves them alone. The risk is the other one:
    a member added to the vocabulary and not to a table keyed by it, which
    surfaces as a `KeyError` on whichever input reaches the gap first, or
    worse as a silently skipped case.

    Each pair below is measured equal today. The assertion is equality rather
    than containment so growth on either side is caught, in whichever module
    it happens.
    """

    @staticmethod
    def _pairs():
        from uacpy.acoustic_signal.arrays import _RADON_KINDS
        from uacpy.acoustic_signal.estimate import _SPECTRAL_SCALINGS
        from uacpy.core.results.rays import _BOUNCE_KINDS
        from uacpy.core.source import VALID_SOURCE_TYPES
        from uacpy.data.sound_speed import _GRIDS
        from uacpy.data.woa23_local import _CODE
        from uacpy.io.bellhop_writer import _VALID_BEAM_TYPES
        from uacpy.io.oalib_writer import SOURCE_TYPE_CODE
        from uacpy.models.bellhop import (_BEAM_TYPE_RUN_TYPES,
                                          _INFLUENCE_ROUTINE)
        from uacpy.visualization.plots.rays_modes import RAY_CLASS_COLOURS
        from uacpy.visualization.plots.signal import (
            _HISTOGRAM_KIND, _RADON_AXIS, _SCALING_KIND, _SCALING_UNIT)
        return [
            ('rays._BOUNCE_KINDS', _BOUNCE_KINDS,
             'rays_modes.RAY_CLASS_COLOURS', RAY_CLASS_COLOURS),
            ('source.VALID_SOURCE_TYPES', VALID_SOURCE_TYPES,
             'oalib_writer.SOURCE_TYPE_CODE', SOURCE_TYPE_CODE),
            ('estimate._SPECTRAL_SCALINGS', _SPECTRAL_SCALINGS,
             'signal._SCALING_UNIT', _SCALING_UNIT),
            ('estimate._SPECTRAL_SCALINGS', _SPECTRAL_SCALINGS,
             'signal._SCALING_KIND', _SCALING_KIND),
            ('estimate._SPECTRAL_SCALINGS', _SPECTRAL_SCALINGS,
             'signal._HISTOGRAM_KIND', _HISTOGRAM_KIND),
            ('arrays._RADON_KINDS', _RADON_KINDS,
             'signal._RADON_AXIS', _RADON_AXIS),
            ('bellhop_writer._VALID_BEAM_TYPES', _VALID_BEAM_TYPES,
             'bellhop._BEAM_TYPE_RUN_TYPES', _BEAM_TYPE_RUN_TYPES),
            ('bellhop_writer._VALID_BEAM_TYPES', _VALID_BEAM_TYPES,
             'bellhop._INFLUENCE_ROUTINE', _INFLUENCE_ROUTINE),
            ('sound_speed._GRIDS', _GRIDS, 'woa23_local._CODE', _CODE),
        ]

    def test_each_table_keys_on_exactly_its_vocabulary(self):
        mismatched = []
        for vocab_name, vocab, table_name, table in self._pairs():
            stray = set(table) - set(vocab)
            missing = set(vocab) - set(table)
            if stray or missing:
                mismatched.append(
                    f'{table_name} vs {vocab_name}: '
                    f'stray={sorted(stray)} missing={sorted(missing)}')
        assert not mismatched, '\n  '.join([''] + mismatched)


class TestEveryCollapseMethodReachesAnImplementation:
    """The model layer's collapse registry and the carriers must agree.

    ``PropagationModel.__init__`` validates ``collapse=`` against
    ``VALID_COLLAPSE_METHODS`` so a bad method fails at construction rather
    than from inside a writer at ``run()``-time. That promise holds only
    while every method the registry lists is one the carrier actually
    implements: a method accepted at construction and refused by the carrier
    raises from a class the user never named, after the run has started.

    ``VALID_COLLAPSE_METHODS`` is now built from the carriers' own
    vocabularies, so the two sides cannot disagree by construction. This
    gate covers the half that construction cannot: that each carrier's
    *implementation* still accepts every method its vocabulary names. A
    branch deleted from a ladder is invisible to the registry.
    """

    @staticmethod
    def _range_dependent_ssp():
        from uacpy.core.ssp import SoundSpeedProfile
        return SoundSpeedProfile(depths=[0.0, 100.0],
                                 data=[[1500.0, 1510.0], [1490.0, 1495.0]],
                                 ranges=[0.0, 1000.0])

    @staticmethod
    def _range_dependent_surface():
        from uacpy.core.bottom import BoundaryProperties
        from uacpy.core.surface import Surface
        return Surface(properties=[
            BoundaryProperties(acoustic_type='half-space', sound_speed=340.0,
                               density=0.0012, attenuation=0.0,
                               roughness=r)
            for r in (0.0, 0.5)],
            ranges=[0.0, 1000.0])

    @staticmethod
    def _range_dependent_bottom():
        from uacpy.core.bottom import Bottom
        # Half-spaces, not layer stacks: 'mean' over a layered bottom is
        # undefined by design (a stack cannot be averaged) and refuses with
        # its own message, which is a property of the *input*, not of the
        # vocabulary this gate measures.
        return Bottom.from_halfspaces(
            ranges=[0.0, 1000.0], sound_speed=[1600.0, 1700.0],
            density=[1.9, 2.0], attenuation=[0.5, 0.6])

    @staticmethod
    def _layered_column():
        from uacpy.core.bottom import (BoundaryProperties, SedimentLayer,
                                       SeabedColumn)
        return SeabedColumn(
            layers=[SedimentLayer(thickness=10.0, sound_speed=1600.0,
                                  density=1.8, attenuation=0.3)],
            halfspace=BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=1800.0, density=2.0,
                                         attenuation=0.5))

    def _cases(self):
        """``(registry key, description, call)`` per carrier-backed key.

        ``'altimetry'`` and ``'elastic'`` are absent on purpose: both are
        implemented inside ``models/base.py`` itself rather than on a
        carrier, so there is no second layer for them to drift from.
        """
        from uacpy.core.environment import Environment
        env = Environment(name='slope',
                          bathymetry=[(0.0, 100.0), (1000.0, 200.0)])
        ssp = self._range_dependent_ssp()
        surface = self._range_dependent_surface()
        bottom = self._range_dependent_bottom()
        column = self._layered_column()
        return [
            ('bathymetry', 'Environment.get_representative_depth',
             env.get_representative_depth),
            ('ssp', 'SoundSpeedProfile.collapse', ssp.collapse),
            ('surface', 'Surface.collapse', surface.collapse),
            ('bottom_range', 'Bottom.select_range',
             lambda m: bottom.select_range(m)),
            ('bottom_layers', 'SeabedColumn.collapse', column.collapse),
        ]

    def test_each_carrier_accepts_every_method_the_registry_lists(self):
        from uacpy.models.base import VALID_COLLAPSE_METHODS
        refused = []
        for key, who, call in self._cases():
            for method in sorted(VALID_COLLAPSE_METHODS[key]):
                try:
                    call(method)
                except Exception as exc:          # noqa: BLE001 — reported
                    refused.append(
                        f"collapse[{key!r}]={method!r} is accepted by "
                        f"PropagationModel but {who} raises "
                        f"{type(exc).__name__}: {exc}")
        assert not refused, '\n  '.join([''] + refused)

    @staticmethod
    def _degenerate_cases():
        """The same carriers with nothing to reduce.

        Every fixture above is range-*dependent*, which is the branch that
        reaches the reducing code — and it is not the branch most callers
        are on. A guard placed after the "nothing to reduce" early return
        accepts any string on a 1-D profile and refuses it on a 2-D one, so
        a typo surfaces only when the user moves to a range-dependent
        environment. That is the case this covers.
        """
        from uacpy.core.bottom import (Bottom, BoundaryProperties,
                                        SeabedColumn)
        from uacpy.core.environment import Environment
        from uacpy.core.ssp import SoundSpeedProfile
        from uacpy.core.surface import Surface
        flat = Environment(name='flat', bathymetry=100.0)
        return [
            ('bathymetry', 'Environment.get_representative_depth',
             flat.get_representative_depth),
            ('ssp', 'SoundSpeedProfile.collapse',
             SoundSpeedProfile(depths=[0.0, 100.0],
                               data=[1500.0, 1480.0]).collapse),
            ('surface', 'Surface.collapse',
             Surface(properties=[
                 BoundaryProperties(acoustic_type='vacuum')]).collapse),
            ('bottom_range', 'Bottom.select_range',
             Bottom.from_halfspaces(ranges=[0.0], sound_speed=[1600.0],
                                    density=[1.9],
                                    attenuation=[0.5]).select_range),
            # A column with no layers is a pure half-space: the degenerate
            # case for flattening a stack, and the fifth key, so this gate
            # covers the same five as _cases rather than four of them.
            ('bottom_layers', 'SeabedColumn.collapse',
             SeabedColumn(layers=[], halfspace=BoundaryProperties(
                 acoustic_type='half-space', sound_speed=1800.0,
                 density=2.0, attenuation=0.5)).collapse),
        ]

    def test_a_carrier_with_nothing_to_reduce_refuses_a_stranger(self):
        """A typo must not wait for a range-dependent environment to show up."""
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.models.base import VALID_COLLAPSE_METHODS
        accepted = []
        for key, who, call in self._degenerate_cases():
            stranger = 'not_a_collapse_method'
            assert stranger not in VALID_COLLAPSE_METHODS[key]
            try:
                call(stranger)
            except ConfigurationError:
                continue
            accepted.append(f"{who} accepted {stranger!r} when there was "
                            f"nothing to reduce")
        assert not accepted, '\n  '.join([''] + accepted)

    def test_a_carrier_with_nothing_to_reduce_accepts_each_listed_method(self):
        """And the guard must not have overshot into refusing real ones."""
        from uacpy.models.base import VALID_COLLAPSE_METHODS
        refused = []
        for key, who, call in self._degenerate_cases():
            for method in sorted(VALID_COLLAPSE_METHODS[key]):
                try:
                    call(method)
                except Exception as exc:      # noqa: BLE001 — reported
                    refused.append(f"{who}({method!r}) raised "
                                   f"{type(exc).__name__}: {exc}")
        assert not refused, '\n  '.join([''] + refused)

    def test_each_carrier_refuses_a_method_the_registry_does_not_list(self):
        """Silence above must mean "accepted", never "accepts anything"."""
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.models.base import VALID_COLLAPSE_METHODS
        accepted = []
        for key, who, call in self._cases():
            stranger = 'not_a_collapse_method'
            assert stranger not in VALID_COLLAPSE_METHODS[key]
            try:
                call(stranger)
            except ConfigurationError:
                continue
            accepted.append(f"{who} accepted {stranger!r}")
        assert not accepted, '\n  '.join([''] + accepted)


class TestEveryResultTypeHasExactlyOnePlotterRow:
    """``_PLOTTERS`` must cover the ``Result`` hierarchy, and only it.

    ``Result.plot`` is the documented way to draw any result, so a subclass
    missing from the table is a result that cannot be drawn at all — and the
    failure surfaces as "no plotter registered", from a call the user made
    on the object itself.

    The row also carries ``draws_env``. That used to be a second tuple
    inside ``plot_result``, listing the complement of the branches that
    accept ``env=``; a type added to one and not the other left ``env=``
    accepted and silently ignored. One row per type makes the pair
    inseparable, and this gate makes the row itself mandatory.
    """

    @staticmethod
    def _concrete_result_types():
        """Every importable ``Result`` subclass except the base and the
        stack. ``ResultStack`` dispatches on the type of the slabs it holds,
        so ``plot_result`` handles it before the table."""
        import uacpy.core.results as results_pkg
        from uacpy.core.results._base import Result
        from uacpy.core.results import ResultStack
        found = {}
        for name in dir(results_pkg):
            obj = getattr(results_pkg, name)
            if (isinstance(obj, type) and issubclass(obj, Result)
                    and obj not in (Result, ResultStack)):
                found[obj] = name
        return found

    def test_the_table_lists_every_result_subclass_exactly_once(self):
        from uacpy.visualization.plots import _PLOTTERS
        listed = [row[0] for row in _PLOTTERS]
        found = self._concrete_result_types()
        # Silence must mean "they match", never "nothing was discovered".
        assert len(found) >= 7, (
            f'only {len(found)} Result subclass(es) discovered — the '
            f'discovery walk has stopped seeing them: {sorted(found.values())}')
        missing = sorted(n for c, n in found.items() if c not in listed)
        stray = sorted(c.__name__ for c in listed if c not in found)
        duplicated = sorted(c.__name__ for c in set(listed)
                            if listed.count(c) > 1)
        assert not (missing or stray or duplicated), (
            f'_PLOTTERS vs the Result hierarchy: missing={missing} '
            f'stray={stray} duplicated={duplicated}')

    def test_each_row_names_a_plotter_that_takes_env_iff_it_draws_one(self):
        """``draws_env`` is checked against the plotter's own signature, so
        a row cannot promise an environment the plotter cannot accept."""
        import inspect
        from uacpy.visualization.plots import _PLOTTERS
        wrong = []
        for result_type, plotter, draws_env in _PLOTTERS:
            params = inspect.signature(plotter).parameters
            takes_env = 'env' in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in params.values())
            if draws_env and not takes_env:
                wrong.append(f'{result_type.__name__}: row says it draws an '
                             f'environment but {plotter.__name__} has no '
                             f'env parameter')
            # The other direction, which the "iff" in the name promises: a
            # plotter that names env= while its row says False would have
            # env= refused for it, so the refusal and the capability part
            # company with nothing to say so.
            if not draws_env and 'env' in params:
                wrong.append(f'{result_type.__name__}: row says it draws no '
                             f'environment but {plotter.__name__} names an '
                             f'env parameter, so env= is refused for a '
                             f'plotter that accepts it')
        assert not wrong, '\n  '.join([''] + wrong)


class TestEveryResultCarrierCanDrawItself:
    """``result = compute(...)`` then ``fig, ax = result.plot()``, everywhere.

    uacpy returns a named carrier from every measurement, and most of them
    grew a ``.plot()``; nine did not, so ``welch(x, fs).plot()`` worked while
    ``spectrogram(x, fs).plot()`` did not, and the reader had to know which
    of 56 plotters took which arrays in which order. The plotter still
    exists and still takes arrays — ``.plot()`` is the one obvious way, not
    the only way.

    This gate is the part that outlives the fix: a carrier added later
    cannot ship without either a ``.plot()`` or a line in
    ``_NO_PLOTTER_YET`` saying why it has none.
    """

    #: Carriers with no plotter in the package, and the reason. A name here
    #: is a statement that nothing draws it, not that nothing should.
    _NO_PLOTTER_YET = {
        'BeamformResult': 'scalar detection summary (snr, angles, peak_snr); '
                          'plot_beam_power draws a BeamformedField, which is '
                          'a different carrier and has its own .plot()',
        'NoiseComponents': 'a view of the five WenzNoise terms; WenzNoise '
                           'itself plots them, together or alone',
        'DataProvenance': 'attribution metadata, not a measurement',
        'Snapshots': 'an intermediate on the way to a covariance; what you '
                     'draw is the covariance or the beam surface it feeds, '
                     'and neither is this carrier',
    }

    @staticmethod
    def _public_carriers():
        """Every public named-tuple carrier and ``Result`` subclass.

        Structural, not by name: a carrier is a public class that is either
        a ``namedtuple`` (it has ``_fields``) or a ``Result`` subclass. A
        name-based rule would miss ``ComplexCepstrum`` and ``ChannelTaps``,
        neither of which is called a Result.
        """
        import importlib
        import inspect
        from uacpy.core.results._base import Result
        found = {}
        for mod_name in ('uacpy', 'uacpy.acoustic_signal', 'uacpy.comms',
                         'uacpy.noise', 'uacpy.sonar', 'uacpy.core.results'):
            mod = importlib.import_module(mod_name)
            for attr in getattr(mod, '__all__', ()):
                obj = getattr(mod, attr, None)
                if not inspect.isclass(obj):
                    continue
                is_carrier = (issubclass(obj, tuple)
                              and hasattr(obj, '_fields')) or (
                                  issubclass(obj, Result) and obj is not Result)
                if is_carrier:
                    found[attr] = obj
        return found

    def test_the_discovery_walk_finds_the_known_carriers(self):
        """Silence must mean "they all plot", never "none were found"."""
        found = self._public_carriers()
        assert len(found) >= 15, sorted(found)
        for expect in ('SpectralEstimate', 'FKResult', 'ComplexCepstrum',
                       'Field', 'ChannelTaps'):
            assert expect in found, (expect, sorted(found))

    def test_each_one_has_a_plot_method(self):
        missing = [name for name, cls in self._public_carriers().items()
                   if not hasattr(cls, 'plot')
                   and name not in self._NO_PLOTTER_YET]
        assert not missing, (
            f'result carrier(s) with no .plot(): {sorted(missing)}. Add one '
            f'delegating to the plotter that draws it, or record in '
            f'_NO_PLOTTER_YET why nothing does.')

    def test_the_exemptions_are_real(self):
        """An exemption naming a carrier that *does* plot is stale, and an
        exemption for a name that no longer exists is worse — it would
        silently excuse a future carrier that reused the name."""
        carriers = self._public_carriers()
        for name in self._NO_PLOTTER_YET:
            if name in carriers:
                assert not hasattr(carriers[name], 'plot'), (
                    f'{name} has a .plot() but is listed as having no '
                    f'plotter — drop it from _NO_PLOTTER_YET')

    def test_plot_returns_a_figure_and_an_axis(self):
        """The shape the whole package promises, measured on one carrier of
        each construction: a plain namedtuple subclass, a namedtuple with
        attributes, and a non-tuple carrier."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        import uacpy
        from uacpy import acoustic_signal as sig

        fs = 2000.0
        x = np.sin(2 * np.pi * 200 * np.arange(1024) / fs)
        panel = np.stack([np.roll(x, 3 * i) for i in range(8)], axis=1)
        for carrier in (sig.spectrogram(x, fs),
                        sig.fk_transform(panel, fs, 1.0),
                        uacpy.absorption_thorp(np.array([1e2, 1e3, 1e4]))):
            out = carrier.plot()
            try:
                assert isinstance(out, tuple) and len(out) == 2, (
                    f'{type(carrier).__name__}.plot() returned {out!r}')
            finally:
                plt.close('all')


class TestBothSpellingsOfAPlotAgree:
    """``carrier.plot()`` and ``plot_x(carrier)`` must be the same call.

    ``.plot()`` is the one obvious way; the free plotter is the one that
    still takes bare arrays, for numbers that did not come from uacpy. Both
    have to exist and they have to draw the same thing, or the convenience
    method becomes a second, subtly different renderer.

    Before this, only four of thirteen plotters accepted their own carrier:
    the rest declared the other arrays as required positionals, so
    ``plot_spectrogram(result)`` raised ``TypeError`` from the signature
    before any code ran.
    """

    @staticmethod
    def _cases():
        import numpy as np
        import uacpy
        from uacpy import acoustic_signal as sig, noise
        fs = 2000.0
        x = np.sin(2 * np.pi * 200 * np.arange(1024) / fs)
        panel = np.stack([np.roll(x, 3 * i) for i in range(8)], axis=1)
        f = np.array([100.0, 1000.0, 10000.0])
        moveout = np.linspace(-1e-3, 1e-3, 21)
        return [
            (sig.welch(x, fs), 'plot_psd', {}),
            (sig.spectrogram(x, fs), 'plot_spectrogram', {}),
            (sig.cwt(x, fs), 'plot_cwt', {'sample_rate': fs}),
            (sig.wigner_ville(x[:256], fs), 'plot_wigner_ville', {}),
            (sig.complex_cepstrum(x), 'plot_cepstrum', {}),
            (sig.constant_q_transform(x, fs), 'plot_constant_q_transform', {}),
            (sig.constant_q_spectrogram(x, fs),
             'plot_constant_q_spectrogram', {}),
            (sig.ambiguity_function(x[:256], fs), 'plot_ambiguity', {}),
            (sig.fk_transform(panel, fs, 1.0), 'plot_fk', {}),
            (sig.taup_transform(panel, fs, 1.0), 'plot_taup', {}),
            (sig.radon_transform(panel, fs, 1.0, moveout), 'plot_radon', {}),
            (uacpy.absorption_thorp(f), 'plot_absorption', {}),
            (noise.WenzNoise(f, wind_speed_kn=15.0), 'plot_wenz', {}),
        ]

    @staticmethod
    def _drawn(ax):
        """What an axis actually has on it, cheaply comparable.

        Lines, images and mesh collections, because the family draws all
        three: comparing only ``ax.lines`` would call two heatmaps equal
        whatever they contain, which is the failure this gate exists to
        catch.
        """
        import numpy as np
        lines = [tuple(np.asarray(ln.get_ydata()).ravel()[:8])
                 for ln in ax.lines]
        images = [float(np.nansum(im.get_array())) for im in ax.images]
        meshes = [float(np.nansum(c.get_array())) for c in ax.collections
                  if getattr(c, 'get_array', None) is not None
                  and c.get_array() is not None]
        return lines, images, meshes

    def test_every_plotter_accepts_its_own_carrier(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from uacpy import visualization
        refused = []
        cases = self._cases()
        assert len(cases) >= 13, len(cases)
        for carrier, plotter, kwargs in cases:
            try:
                getattr(visualization, plotter)(carrier, **kwargs)
            except Exception as exc:                  # noqa: BLE001 reported
                refused.append(f'{plotter}({type(carrier).__name__}) raised '
                               f'{type(exc).__name__}: {exc}')
            finally:
                plt.close('all')
        assert not refused, '\n  '.join([''] + refused)

    def test_the_two_spellings_draw_the_same_thing(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from uacpy import visualization
        differ = []
        for carrier, plotter, kwargs in self._cases():
            try:
                _f1, ax1 = getattr(visualization, plotter)(carrier, **kwargs)
                _f2, ax2 = carrier.plot(**kwargs)
                if self._drawn(ax1) != self._drawn(ax2):
                    differ.append(f'{type(carrier).__name__}: {plotter}'
                                  f'(carrier) and .plot() drew differently')
            finally:
                plt.close('all')
        assert not differ, '\n  '.join([''] + differ)

    def test_a_carrier_with_real_arrays_beside_it_is_not_unpacked(self):
        """The only input the "others must be None" guard decides.

        Passing three ndarrays is settled one line later by the ``_fields``
        test, so it re-tests that instead. This passes a *namedtuple* in the
        first slot with real arrays in the others — without the guard the
        carrier wins and the two explicit arrays are silently discarded.
        """
        import collections
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from uacpy import acoustic_signal as sig, visualization
        result = sig.spectrogram(
            np.sin(2 * np.pi * 200 * np.arange(1024) / 2000.0), 2000.0)
        # A namedtuple whose leading fields match, so only the guard can
        # tell this call from the carrier form.
        Triple = collections.namedtuple(
            'SpectrogramResult', 'frequencies times power')
        # Same shapes, different values, so only which one was READ can
        # separate the two calls.
        decoy = Triple(result.frequencies, result.times, result.power * 100.0)
        from uacpy.core.exceptions import ConfigurationError
        try:
            # Guarded, the carrier is left alone, so `frequencies` is still a
            # 3-tuple and the call fails loudly. Unguarded, the carrier wins
            # and the call SUCCEEDS while quietly drawing the decoy's power
            # instead of the array handed to it — a wrong figure, not an
            # error. Refusal here is the pin.
            with pytest.raises(ConfigurationError):
                visualization.plot_spectrogram(
                    decoy, result.times, result.power)
            # The carrier form alone still works, so the guard has not
            # simply disabled the feature.
            _fig, ax = visualization.plot_spectrogram(decoy)
            assert self._drawn(ax)
        finally:
            plt.close('all')

    def test_an_array_call_is_never_reinterpreted(self):
        """The carrier form is recognised only when the other positional
        arguments are None, so passing arrays still means arrays."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from uacpy import acoustic_signal as sig, visualization
        result = sig.spectrogram(
            np.sin(2 * np.pi * 200 * np.arange(1024) / 2000.0), 2000.0)
        try:
            _f1, ax1 = visualization.plot_spectrogram(
                result.frequencies, result.times, result.power)
            _f2, ax2 = visualization.plot_spectrogram(result)
            assert self._drawn(ax1) == self._drawn(ax2)
        finally:
            plt.close('all')

    def test_a_plain_tuple_of_data_is_data_not_a_carrier(self):
        """The detection keys on ``_fields``, not on ``isinstance(tuple)``.

        ``plot_cepstrum(tuple_of_64_floats)`` is a legitimate array call with
        no other argument to disambiguate it. Reading any tuple as a carrier
        drew its first element as a one-point line — a silent wrong answer,
        not an error.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from uacpy import visualization
        values = tuple(float(v) for v in np.sin(np.arange(64) / 5.0))
        try:
            _fig, ax = visualization.plot_cepstrum(values)
            assert len(ax.lines[0].get_ydata()) == len(values)
        finally:
            plt.close('all')

    def test_an_incomplete_array_call_is_refused_by_name(self):
        """Giving the arrays None defaults must not turn a missing argument
        into a crash inside matplotlib."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from uacpy import visualization
        from uacpy.core.exceptions import ConfigurationError
        try:
            with pytest.raises(ConfigurationError, match='or one'):
                visualization.plot_spectrogram(np.arange(4), np.arange(3))
            with pytest.raises(ConfigurationError, match='or one'):
                visualization.plot_taup(np.arange(4))
        finally:
            plt.close('all')


class TestChannelTapsDrawsOnTheAxesItCarries:
    """Both of ``plot_channel``'s axes come from the carrier, not the index.

    ``ChannelTaps`` holds ``symbol_rate`` and ``sps`` but no sample rate, so
    the rate is derived as ``symbol_rate * sps``; mutating that to
    ``symbol_rate`` alone mislabels the frequency axis by a factor of ``sps``
    — 4 to 8 in practice — and left every suite green.

    The delay axis is ``delays_s``, and the same class of mistake lives there:
    ``arange(n) / fs`` starts at zero, while the tap grid starts ``span/2``
    symbols EARLIER than the first arrival, on the leading skirt of the
    transmit pulse. On a span-8 channel at 50 Bd that draws the first arrival
    at +80 ms with nothing on the figure to say so — and a reader who crops
    to the first few symbols sees an empty axis with the taps climbing off
    the right-hand edge.
    """

    SPS, SYMBOL_RATE = 4, 500.0

    def _taps(self):
        import numpy as np
        from uacpy.core.results.rays import ChannelTaps
        # The peak is NOT the first sample and the grid starts negative, so
        # an index axis and the carried one disagree about where the peak is.
        return ChannelTaps(taps=np.array([0.4j, 1.0, -0.2]),
                           delays_s=np.array([-1.0, 0.0, 1.0]) / (
                               self.SYMBOL_RATE * self.SPS),
                           symbol_rate=self.SYMBOL_RATE, carrier=12000.0,
                           sps=self.SPS, first_arrival_s=0.25)

    @staticmethod
    def _delay_axis(axes):
        """The delay panel's drawn x, in ms. ``stem`` puts the markers on a
        Line2D and the stems on a collection, so reading the first line is
        enough and is what a reader of the figure sees."""
        import numpy as np
        return np.asarray(axes[0].lines[0].get_xdata(), dtype=float)

    def test_the_peak_tap_is_drawn_at_the_delay_it_carries(self):
        """``delays_s`` places the taps. With an index axis the peak here
        lands at +0.5 ms instead of 0, and every tap with it."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        import pytest as _pytest
        taps = self._taps()
        try:
            _fig, axes = taps.plot()
            drawn = self._delay_axis(axes)
        finally:
            plt.close('all')
        peak = int(np.argmax(np.abs(taps.taps)))
        assert drawn[peak] == _pytest.approx(0.0, abs=1e-9), (
            f'the loudest tap is drawn at {drawn[peak]:g} ms; delays_s puts '
            f'it at {taps.delays_s[peak] * 1e3:g} ms')
        assert drawn[0] == _pytest.approx(taps.delays_s[0] * 1e3), (
            f'the grid starts at {drawn[0]:g} ms; delays_s starts at '
            f'{taps.delays_s[0] * 1e3:g} ms')

    def test_the_free_plotter_and_the_method_draw_the_same_axis(self):
        """``plot_channel(taps)`` is the call ``taps.plot()`` makes, so the
        two have to place the taps identically. Passing bare arrays keeps the
        index axis, which is the form for taps with no delay reference."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from uacpy import visualization
        taps = self._taps()
        try:
            free = self._delay_axis(visualization.plot_channel(taps)[1])
            method = self._delay_axis(taps.plot()[1])
            bare = self._delay_axis(
                visualization.plot_channel(taps.taps,
                                           self.SYMBOL_RATE * self.SPS)[1])
        finally:
            plt.close('all')
        assert np.allclose(free, method), (free, method)
        assert np.allclose(bare, np.arange(taps.taps.size) * 1e3 / (
            self.SYMBOL_RATE * self.SPS)), bare

    def test_the_frequency_axis_spans_the_derived_sample_rate(self):
        """The right-hand panel is two-sided, so its axis runs to ±fs/2 with
        fs = symbol_rate * sps. Reading the axis is what catches a rate that
        is wrong by a factor of sps; reading the taps would not."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pytest as _pytest
        taps = self._taps()
        try:
            _fig, axes = taps.plot()
            drawn = axes[1].lines[0].get_xdata()
        finally:
            plt.close('all')
        expected_fs = self.SYMBOL_RATE * self.SPS
        assert max(abs(drawn)) == _pytest.approx(expected_fs / 2.0,
                                                 rel=1e-3), (
            f'frequency axis reaches {max(abs(drawn)):g} Hz; a two-sided '
            f'spectrum at symbol_rate*sps = {expected_fs:g} Hz should reach '
            f'{expected_fs / 2.0:g}')

    def test_it_returns_the_pair_of_axes_its_plotter_makes(self):
        """``plot_channel`` draws delay and frequency panels, so this one
        method returns two axes where the rest of the family returns one."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        try:
            fig, axes = self._taps().plot()
            assert np.size(axes) == 2, axes
        finally:
            plt.close('all')


class TestAPlotterDrawsOnlyItsOwnCarrier:
    """Every carrier against every plotter, not a sample.

    Giving the array parameters ``None`` defaults removed a type check the
    required positionals used to provide for free, and the first replacement
    compared ``_fields`` only — which cannot separate ``SpectrogramResult``
    from ``CQSpectrogramResult``, both ``(frequencies, times, power)``.
    Drawing either with the other's plotter is a unit error in both
    directions: a density comes out labelled ``Pa²``, geometric bins get a
    linear axis, and linear bins get a log axis whose first bin is 0 Hz.

    Sampling a few pairs is what let that through, so this sweeps the whole
    matrix. `plot_fk` is in it because it kept a separate inline detection
    and so missed the first fix entirely.
    """

    @staticmethod
    def _matrix():
        import numpy as np
        from uacpy import acoustic_signal as sig
        fs = 2000.0
        x = np.sin(2 * np.pi * 200 * np.arange(2048) / fs)
        panel = np.stack([np.roll(x, 3 * i) for i in range(8)], axis=1)
        moveout = np.linspace(-1e-3, 1e-3, 21)
        carriers = {
            'SpectrogramResult': sig.spectrogram(x, fs),
            'CQSpectrogramResult': sig.constant_q_spectrogram(x, fs),
            'WignerVilleResult': sig.wigner_ville(x[:256], fs),
            'TauPResult': sig.taup_transform(panel, fs, 1.0),
            'RadonResult': sig.radon_transform(panel, fs, 1.0, moveout),
            'FKResult': sig.fk_transform(panel, fs, 1.0),
            'AmbiguityResult': sig.ambiguity_function(x[:256], fs),
            'CQTResult': sig.constant_q_transform(x, fs),
            'CWTResult': sig.cwt(x, fs),
            'ComplexCepstrum': sig.complex_cepstrum(x),
        }
        owner = {
            'plot_spectrogram': 'SpectrogramResult',
            'plot_constant_q_spectrogram': 'CQSpectrogramResult',
            'plot_wigner_ville': 'WignerVilleResult',
            'plot_taup': 'TauPResult',
            'plot_radon': 'RadonResult',
            'plot_fk': 'FKResult',
            'plot_ambiguity': 'AmbiguityResult',
            'plot_constant_q_transform': 'CQTResult',
            'plot_cwt': 'CWTResult',
            'plot_cepstrum': 'ComplexCepstrum',
        }
        extra = {'plot_cwt': {'sample_rate': fs},
                 'plot_fk': {'scaling': 'power'}}
        return carriers, owner, extra

    def test_each_plotter_accepts_its_own_carrier_and_no_other(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from uacpy import visualization
        carriers, owner, extra = self._matrix()
        assert len(carriers) == 10 and len(owner) == 10, (
            'the matrix has stopped covering the family')
        from uacpy.core.exceptions import ConfigurationError
        accepted_foreign, refused_own, refused_by_luck = [], [], []
        for plotter, own in owner.items():
            fn = getattr(visualization, plotter)
            for name, carrier in carriers.items():
                why = None
                try:
                    fn(carrier, **extra.get(plotter, {}))
                    drew = True
                except ConfigurationError as exc:
                    drew, why = False, str(exc)
                except Exception as exc:          # noqa: BLE001 — classified
                    drew, why = False, f'{type(exc).__name__}: {exc}'
                finally:
                    plt.close('all')
                if name == own:
                    if not drew:
                        refused_own.append(f'{plotter}({name}): {why}')
                    continue
                if drew:
                    accepted_foreign.append(f'{plotter}({name})')
                # Refused BY THE CARRIER GUARD, not by luck further down.
                # `plot_ambiguity(SpectrogramResult)` was already refused by
                # a grid-shape accident before any type check existed, so
                # counting "did not draw" as a pass would leave that cell
                # green with the guard deleted. Either of the guard's two
                # messages counts — the width branch ("needs N") and the
                # identity branch ("that <plotter> draws") are both it doing
                # its job; anything else is downstream.
                elif not any(phrase in (why or '') for phrase in
                             (f'that {plotter} draws',
                              f'{plotter} needs')):
                    refused_by_luck.append(f'{plotter}({name}): {why}')
        assert not accepted_foreign, (
            'plotter(s) drew a carrier that is not theirs: '
            + ', '.join(accepted_foreign))
        assert not refused_own, (
            'plotter(s) refused their own carrier:\n  '
            + '\n  '.join(refused_own))
        assert not refused_by_luck, (
            'foreign carrier(s) refused for a reason other than the type '
            'check, so these cells would stay green without it:\n  '
            + '\n  '.join(refused_by_luck))
