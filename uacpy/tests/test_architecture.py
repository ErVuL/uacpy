"""Rules about the SHAPE of the package, not the value of any number.

Every rule here was a defect first. Each one was found by reading, once, and
fixed by hand; these exist so the next instance is found by the suite instead.
They are deliberately narrow: a rule that cannot say precisely what is wrong
is a rule that gets suppressed.

* :func:`test_no_plotter_branches_on_a_model_name` — a plotter draws what it
  is handed. ``plot_absorption(frequencies, model='thorp')`` both selected a
  formula and drew it, which made it a second spelling of
  ``Thorp().table(f).plot()``; the name-keyed map inside it had no other
  caller in the package, so the map had no reason to exist either.
* :func:`test_no_vocabulary_is_declared_twice` — ``LOSS_KINDS`` was declared
  in the data layer and again in the plotting layer, beside a comment
  asserting it was declared once. A third kind added to one would have made
  ``Field.max`` report the loud end while the 1-D cut drew its axis the other
  way.
* :func:`test_a_decibel_is_never_spelled_db` — ``db`` is a database and ``dB``
  is a decibel. Two identifiers broke it (``vmin_db``, ``diag_db``); the ten
  that keep it are real databases and are listed.
* :func:`test_every_refusal_pin_names_its_refusal` — 369 ``pytest.raises``
  calls pinned the exception type alone, which any refusal of that type
  satisfies: a test built to reach one refusal stayed green when another,
  earlier one fired instead.
"""
import ast
import builtins
import collections
from pathlib import Path

import pytest

import uacpy
import uacpy.core.exceptions as _exceptions
import numpy as np
from uacpy.tests._internal_modules import (PACKAGE_INTERNAL_MODULES,
                                           private_module)

#: Every test here pins a repo convention (sources, docs, packaging), not
#: runtime behaviour: ``-m "not convention"`` deselects the module.
pytestmark = pytest.mark.convention

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

    def test_modal_pressure_field_source_density_changes_the_answer(self):
        """A knob that did nothing would be worse than an absent one."""
        import numpy as _np
        kw = dict(source_depth=50.0, receiver_depths=_np.array([50.0]),
                  ranges=_np.array([1000.0]))
        modes = self._modes()
        light = float(_np.asarray(
            modes.modal_pressure_field(**kw, source_density=1.0).tl)[0, 0])
        heavy = float(_np.asarray(
            modes.modal_pressure_field(**kw, source_density=2.0).tl)[0, 0])
        assert heavy > light + 1.0, (light, heavy)

    def test_the_alternative_and_cosmetic_knobs_all_execute(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as _np
        import uacpy
        from uacpy import acoustic_signal
        from uacpy.core.results import ambiguity_field

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
            'plot_mode_excitation(dynamic_range_dB=)': lambda: uacpy.plot.plot_mode_excitation(
                modes, src, sound_speed=1500.0, dynamic_range_dB=30.0),
            'plot_mode_excitation(show_array_factor=)':
                lambda: uacpy.plot.plot_mode_excitation(
                    modes, src, sound_speed=1500.0, show_array_factor=False),
            'plot_matched_field(mark_peak=)': lambda: uacpy.plot.plot_matched_field(
                ambiguity_field(
                    _np.abs(rng.normal(size=(15, 20))) + 1e-6,
                    {'depth': _np.linspace(0.0, 100.0, 15),
                     'range': _np.linspace(0.0, 5e3, 20)},
                    reference_unit='1'),
                mark_peak=False),
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
        from uacpy.acoustic_signal.gathers import _RADON_KINDS
        from uacpy.acoustic_signal.spectral import _SPECTRAL_SCALINGS
        from uacpy.core.results.rays import _BOUNCE_KINDS
        from uacpy.core.source import VALID_SOURCE_TYPES
        from uacpy.data.sound_speed import _GRIDS
        from uacpy.data.woa23_local import _CODE
        from uacpy.io.bellhop_writer import _VALID_BEAM_TYPES
        from uacpy.io.oalib_writer import SOURCE_TYPE_CODE
        from uacpy.models.bellhop._tables import (_BEAM_TYPE_RUN_TYPES,
                                                  _INFLUENCE_ROUTINE)
        from uacpy.visualization.plots.rays_modes import RAY_CLASS_COLOURS
        from uacpy.acoustic_signal._results import POWER_UNITS
        from uacpy.visualization.plots.signal import (
            _HISTOGRAM_KIND, _RADON_AXIS, _SCALING_KIND)
        return [
            ('rays._BOUNCE_KINDS', _BOUNCE_KINDS,
             'rays_modes.RAY_CLASS_COLOURS', RAY_CLASS_COLOURS),
            ('source.VALID_SOURCE_TYPES', VALID_SOURCE_TYPES,
             'oalib_writer.SOURCE_TYPE_CODE', SOURCE_TYPE_CODE),
            ('estimate._SPECTRAL_SCALINGS', _SPECTRAL_SCALINGS,
             '_results.POWER_UNITS', POWER_UNITS),
            ('estimate._SPECTRAL_SCALINGS', _SPECTRAL_SCALINGS,
             'signal._SCALING_KIND', _SCALING_KIND),
            ('estimate._SPECTRAL_SCALINGS', _SPECTRAL_SCALINGS,
             'signal._HISTOGRAM_KIND', _HISTOGRAM_KIND),
            ('arrays._RADON_KINDS', _RADON_KINDS,
             'signal._RADON_AXIS', _RADON_AXIS),
            ('bellhop_writer._VALID_BEAM_TYPES', _VALID_BEAM_TYPES,
             'bellhop._tables._BEAM_TYPE_RUN_TYPES', _BEAM_TYPE_RUN_TYPES),
            ('bellhop_writer._VALID_BEAM_TYPES', _VALID_BEAM_TYPES,
             'bellhop._tables._INFLUENCE_ROUTINE', _INFLUENCE_ROUTINE),
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
                                 sound_speed=[[1500.0, 1510.0], [1490.0, 1495.0]],
                                 ranges=[0.0, 1000.0])

    @staticmethod
    def _range_dependent_surface():
        from uacpy.core.boundary import BoundaryProperties
        from uacpy.core.surface import Surface
        return Surface(nodes=[
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
        from uacpy.core.boundary import BoundaryProperties, SedimentLayer
        from uacpy.core.bottom import SeabedColumn
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
            ('bathymetry', 'Bathymetry.collapse_range',
             env.bathymetry.collapse_range),
            ('ssp', 'SoundSpeedProfile.collapse_range', ssp.collapse_range),
            ('surface', 'Surface.collapse_range', surface.collapse_range),
            ('bottom_range', 'Bottom.collapse_range',
             lambda m: bottom.collapse_range(m)),
            ('bottom_layers', 'SeabedColumn.collapse_layers', column.collapse_layers),
        ]

    def test_each_carrier_accepts_every_method_the_registry_lists(self):
        from uacpy.models._projection import VALID_COLLAPSE_METHODS
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
        from uacpy.core.boundary import BoundaryProperties
        from uacpy.core.bottom import Bottom, SeabedColumn
        from uacpy.core.environment import Environment
        from uacpy.core.ssp import SoundSpeedProfile
        from uacpy.core.surface import Surface
        flat = Environment(name='flat', bathymetry=100.0)
        return [
            ('bathymetry', 'Bathymetry.collapse_range',
             flat.bathymetry.collapse_range),
            ('ssp', 'SoundSpeedProfile.collapse_range',
             SoundSpeedProfile(depths=[0.0, 100.0],
                               sound_speed=[1500.0, 1480.0]).collapse_range),
            ('surface', 'Surface.collapse_range',
             Surface(nodes=[
                 BoundaryProperties(acoustic_type='vacuum')]).collapse_range),
            ('bottom_range', 'Bottom.collapse_range',
             Bottom.from_halfspaces(ranges=[0.0], sound_speed=[1600.0],
                                    density=[1.9],
                                    attenuation=[0.5]).collapse_range),
            # A column with no layers is a pure half-space: the degenerate
            # case for flattening a stack, and the fifth key, so this gate
            # covers the same five as _cases rather than four of them.
            ('bottom_layers', 'SeabedColumn.collapse_layers',
             SeabedColumn(layers=[], halfspace=BoundaryProperties(
                 acoustic_type='half-space', sound_speed=1800.0,
                 density=2.0, attenuation=0.5)).collapse_layers),
        ]

    def test_a_carrier_with_nothing_to_reduce_refuses_a_stranger(self):
        """A typo must not wait for a range-dependent environment to show up."""
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.models._projection import VALID_COLLAPSE_METHODS
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
        from uacpy.models._projection import VALID_COLLAPSE_METHODS
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
        from uacpy.models._projection import VALID_COLLAPSE_METHODS
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
    """``_PLOTTERS`` must cover the ``Result`` hierarchy and the carriers
    that draw themselves, and only them.

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
        from uacpy.core.results import Result
        from uacpy.visualization.plots import _PLOTTERS
        listed = [row[0] for row in _PLOTTERS
                  if issubclass(row[0], Result)]
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
        from uacpy.core.results import Result
        from uacpy.visualization.plots import _PLOTTERS
        wrong = []
        for result_type, plotter, draws_env in _PLOTTERS:
            if not issubclass(result_type, Result):
                continue        # plot_carrier passes no env= at all
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


    def test_the_table_lists_every_carrier_that_draws_itself(self):
        """Every class :mod:`uacpy.core` exports with a ``plot`` method
        that is not a result has one carrier row, so ``.plot()`` on it
        reaches a plotter through :func:`plot_carrier`."""
        import uacpy.core as core
        from uacpy.core.results import Result
        from uacpy.visualization.plots import _PLOTTERS
        from uacpy.core.results import ResultStack
        carriers = {getattr(core, name) for name in core.__all__
                    if isinstance(getattr(core, name), type)
                    and callable(getattr(getattr(core, name), 'plot', None))
                    and not issubclass(getattr(core, name),
                                       (Result, ResultStack))}
        assert len(carriers) >= 5, sorted(c.__name__ for c in carriers)
        rows = [row[0] for row in _PLOTTERS
                if not issubclass(row[0], Result)]
        assert sorted(c.__name__ for c in rows) == sorted(
            c.__name__ for c in carriers)
        assert len(rows) == len(set(rows))


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
        'JanusReception': 'a decode: 64 bits and a CRC verdict; the '
                          'detection statistic it carries is drawn by no '
                          'plotter in the package',
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
                        uacpy.Thorp().table(np.array([1e2, 1e3, 1e4]))):
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
            (sig.cwt(x, fs), 'plot_cwt', {}),
            (sig.wigner_ville(x[:256], fs), 'plot_wigner_ville', {}),
            (sig.complex_cepstrum(x), 'plot_cepstrum', {}),
            (sig.cepstrum(x, sample_rate=fs), 'plot_cepstrum', {}),
            (sig.constant_q_transform(x, fs), 'plot_constant_q_transform', {}),
            (sig.constant_q_spectrogram(x, fs),
             'plot_constant_q_spectrogram', {}),
            (sig.ambiguity_function(x[:256], fs), 'plot_ambiguity', {}),
            (sig.fk_transform(panel, fs, 1.0), 'plot_fk', {}),
            (sig.taup_transform(panel, fs, 1.0), 'plot_taup', {}),
            (sig.radon_transform(panel, fs, 1.0, moveout), 'plot_radon', {}),
            (uacpy.Thorp().table(f), 'plot_absorption', {}),
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
        from uacpy import plot as visualization
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
        from uacpy import plot as visualization
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
        from uacpy import acoustic_signal as sig, plot as visualization
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
            with pytest.raises(ConfigurationError,
                               match='does not match the coordinate lengths'):
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
        from uacpy import acoustic_signal as sig, plot as visualization
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
        from uacpy import plot as visualization
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
        from uacpy import plot as visualization
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
        from uacpy.comms.channel import ChannelTaps
        # The peak is NOT the first sample and the grid starts negative, so
        # an index axis and the carried one disagree about where the peak is.
        return ChannelTaps(taps=np.array([0.4j, 1.0, -0.2]),
                           delays_s=np.array([-1.0, 0.0, 1.0]) / (
                               self.SYMBOL_RATE * self.SPS),
                           symbol_rate=self.SYMBOL_RATE, fc=12000.0,
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
        from uacpy import plot as visualization
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
            'Cepstrum': sig.cepstrum(x, sample_rate=fs),
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
            'plot_cepstrum': ('ComplexCepstrum', 'Cepstrum'),
        }
        extra = {'plot_fk': {'scaling': 'power'}}
        return carriers, owner, extra

    def test_each_plotter_accepts_its_own_carrier_and_no_other(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from uacpy import plot as visualization
        carriers, owner, extra = self._matrix()
        assert len(carriers) == 11 and len(owner) == 10, (
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
                if name in (own if isinstance(own, tuple) else (own,)):
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


# The subpackages whose ``__all__`` lists only functions, classes and
# constants. A submodule listed there reads as a callable in tab completion
# beside the functions (``sonar.target_strength`` sat next to ``ts_sphere``)
# and is still reachable as an attribute without being listed.
_ALL_WITHOUT_SUBMODULES = ('models', 'io', 'acoustic_signal', 'comms', 'data',
                           'sonar', 'noise')


@pytest.mark.parametrize('package', _ALL_WITHOUT_SUBMODULES)
def test_a_subpackage_all_lists_no_submodule(package):
    import importlib
    import types
    module = importlib.import_module(f'uacpy.{package}')
    listed = [name for name in module.__all__
              if isinstance(getattr(module, name, None), types.ModuleType)]
    assert not listed, f'uacpy.{package}.__all__ lists submodule(s) {listed}'


# ── the import graph, read from the source ────────────────────────────────

def _module_name(path):
    """``uacpy/core/ssp.py`` -> ``uacpy.core.ssp``; a package's
    ``__init__.py`` is named by its package."""
    parts = path.relative_to(_PACKAGE.parent).with_suffix('').parts
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


def _module_file(dotted):
    """The source file of the uacpy module ``dotted``, or ``None`` when
    ``dotted`` names no module (a class, a function, a constant)."""
    base = _PACKAGE.parent.joinpath(*dotted.split('.'))
    for candidate in (base.with_suffix('.py'), base / '__init__.py'):
        if candidate.is_file():
            return candidate
    return None


def _uacpy_imports(module, tree):
    """``(kind, target, name, bound, lineno)`` for every import of a uacpy
    module in ``tree``, the source of ``module``.

    ``kind`` is ``'top'`` (runs when the module is imported), ``'lazy'``
    (inside a function body) or ``'tc'`` (under ``if TYPE_CHECKING``).
    ``target`` is the module imported from. ``name`` is the attribute
    imported from it, or ``None`` when the statement imports a module.
    ``bound`` is the local name the statement binds.
    """
    path = _module_file(module)
    is_package = path is not None and path.name == '__init__.py'
    found = []

    def visit(node, kind):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.Lambda)):
                visit(child, 'lazy' if kind == 'top' else kind)
            elif (isinstance(child, ast.If)
                  and (getattr(child.test, 'id', None)
                       or getattr(child.test, 'attr', None))
                  == 'TYPE_CHECKING'):
                for stmt in child.body:
                    visit(ast.Module(body=[stmt], type_ignores=[]), 'tc')
                for stmt in child.orelse:
                    visit(ast.Module(body=[stmt], type_ignores=[]), kind)
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name.split('.')[0] == 'uacpy':
                        found.append((kind, alias.name, None,
                                      alias.asname or 'uacpy', child.lineno))
            elif isinstance(child, ast.ImportFrom):
                target = child.module
                if child.level:
                    here = module.split('.')[:None if is_package else -1]
                    here = here[:len(here) - child.level + 1]
                    target = '.'.join(here + ([child.module]
                                              if child.module else []))
                if not target or target.split('.')[0] != 'uacpy':
                    continue
                for alias in child.names:
                    sub = f'{target}.{alias.name}'
                    found.append((kind,) + (
                        (sub, None) if _module_file(sub) is not None
                        else (target, alias.name))
                        + (alias.asname or alias.name, child.lineno))
            else:
                visit(child, kind)

    visit(tree, 'top')
    return found


def _package_modules():
    """``{module name: parsed source}`` for every first-party module."""
    return {_module_name(path): tree for path, tree in _sources()}


# ── a patch on a module reaches the code that reads the name ──────────────

def _dotted_module(node, bindings):
    """The uacpy module an expression such as ``ram_module`` or
    ``uacpy.models.ram`` names, looked up in ``bindings`` (innermost scope
    first), or ``None``."""
    chain = []
    while isinstance(node, ast.Attribute):
        chain.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    for scope in bindings:
        if node.id in scope:
            dotted = '.'.join([scope[node.id]] + chain[::-1])
            return dotted if _module_file(dotted) is not None else None
    return None


def _module_bindings(module, statements):
    """``{local name: module}`` for the names ``statements`` bind to a
    uacpy module."""
    tree = ast.Module(body=list(statements), type_ignores=[])
    # ``import uacpy.core.ssp`` binds ``uacpy``, the package.
    return {bound: 'uacpy' if bound == 'uacpy' else target
            for _kind, target, name, bound, _line in _uacpy_imports(module,
                                                                    tree)
            if name is None}


def _names_reached(modules):
    """``{(module, name)}`` such that replacing ``name`` on ``module`` changes
    what the code of ``modules`` (``{module name: parsed source}``) sees at
    call time. A function reaches ``module.name`` when it is defined in
    ``module`` and reads ``name`` as a free name, when it imports ``name``
    from ``module`` in its body, or when it reads ``alias.name`` with
    ``alias`` bound to ``module`` in its own body or at module level. A
    module-level ``from module import name`` binds a copy the patch does not
    replace."""
    def imports_in(node):
        return (inner for inner in ast.walk(node)
                if isinstance(inner, (ast.Import, ast.ImportFrom)))

    reached = set()
    for module, tree in modules.items():
        for kind, target, name, _bound, _line in _uacpy_imports(module, tree):
            if kind == 'lazy' and name is not None:
                reached.add((target, name))
        top = {bound: 'uacpy' if bound == 'uacpy' else target
               for kind, target, name, bound, _line
               in _uacpy_imports(module, tree)
               if name is None and kind != 'lazy'}
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.Lambda)):
                continue
            # Two functions may bind one alias to two modules.
            bindings = [_module_bindings(module, imports_in(func)), top]
            for node in ast.walk(func):
                if isinstance(node, ast.Name) and isinstance(node.ctx,
                                                             ast.Load):
                    reached.add((module, node.id))
                elif (isinstance(node, ast.Attribute)
                      and isinstance(node.ctx, ast.Load)):
                    owner = _dotted_module(node.value, bindings)
                    if owner is not None:
                        reached.add((owner, node.attr))
    return reached


def test_a_patch_reaches_the_module_each_function_binds_its_alias_to():
    """Two functions that bind one alias to two modules each reach their own
    (``data/environment.py`` does this for every seabed backend, ``m``); a
    function-level ``from module import name`` reaches ``module.name``; a
    module-level one binds a copy and reaches nothing."""
    source = (
        "from uacpy.data.graw_local import fetch_bottom_graw\n"
        "def mars():\n"
        "    from uacpy.data import mars as m\n"
        "    return m.fetch_bottom_mars\n"
        "def crust1():\n"
        "    from uacpy.data import crust1_local as m\n"
        "    return m.fetch_bottom_crust1\n"
        "def waves():\n"
        "    from uacpy.data.waves import fetch_waves\n"
        "    return fetch_waves\n"
        "def graw():\n"
        "    return fetch_bottom_graw\n")
    reached = _names_reached({'uacpy.probe': ast.parse(source)})
    assert ('uacpy.data.mars', 'fetch_bottom_mars') in reached
    assert ('uacpy.data.crust1_local', 'fetch_bottom_crust1') in reached
    assert ('uacpy.data.mars', 'fetch_bottom_crust1') not in reached
    assert ('uacpy.data.waves', 'fetch_waves') in reached
    assert ('uacpy.data.graw_local', 'fetch_bottom_graw') not in reached


def _module_patches(tree):
    """``(lineno, module, name)`` for every patch in a test module that
    replaces a name on a uacpy MODULE: ``monkeypatch.setattr(mod, 'x', …)``,
    ``monkeypatch.setattr('uacpy.mod.x', …)``, ``patch.object(mod, 'x')``
    and ``patch('uacpy.mod.x')``. Class and instance attributes follow their
    object through a move and are not collected; neither is a target built
    at run time (``importlib.import_module(name)``)."""
    parents = {child: parent for parent in ast.walk(tree)
               for child in ast.iter_child_nodes(parent)}
    found = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not call.args:
            continue
        func = call.func
        verb = (func.attr if isinstance(func, ast.Attribute)
                else getattr(func, 'id', None))
        if verb not in ('setattr', 'object', 'patch'):
            continue
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if verb == 'object' or not first.value.startswith('uacpy.'):
                continue
            module, _, name = first.value.rpartition('.')
            if _module_file(module) is not None:
                found.append((call.lineno, module, name))
            continue
        if verb == 'patch' or len(call.args) < 2:
            continue
        second = call.args[1]
        if not (isinstance(second, ast.Constant)
                and isinstance(second.value, str)):
            continue
        # Imports in the enclosing functions shadow the test module's own.
        scopes, node = [], call
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scopes.append(_module_bindings('uacpy.tests', (
                    inner for inner in ast.walk(node)
                    if isinstance(inner, (ast.Import, ast.ImportFrom)))))
        scopes.append(_module_bindings('uacpy.tests', tree.body))
        module = _dotted_module(first, scopes)
        if module is not None:
            found.append((call.lineno, module, second.value))
    return found


def _unreached_module_patches(test_files, reached):
    """``(patches collected, offenders)`` over ``test_files``."""
    patches, offenders = 0, []
    for path in test_files:
        for lineno, module, name in _module_patches(
                ast.parse(path.read_text(encoding='utf-8'))):
            patches += 1
            if (module, name) not in reached:
                offenders.append(f'{path.name}:{lineno} patches '
                                 f'{module}.{name}')
    return patches, offenders


def test_every_module_patch_reaches_a_function_that_reads_it():
    """``monkeypatch.setattr(module, name, …)`` changes what a function sees
    only when the function looks ``name`` up through ``module`` at call
    time. When the reading code moves to another module and ``module``
    keeps only a re-import, the patch still succeeds, the moved code reads
    the original, and a test that compared two runs or asserted a silence
    passes on the unpatched path. So every module patch in the tests must
    reach a reader (:func:`_names_reached`)."""
    patches, offenders = _unreached_module_patches(
        sorted((_PACKAGE / 'tests').glob('test_*.py')),
        _names_reached(_package_modules()))
    # A resolver that stopped recognising the imports would pass silently.
    assert patches >= 300, f'only {patches} module patches were resolved'
    assert not offenders, (
        'module patches that no package function reads through the patched '
        'module (patch the module whose function reads the name): '
        + ', '.join(offenders))



# ── every warning names its cause ─────────────────────────────────────────

#: The uacpy warning classes a call may name (core/exceptions.py).
_WARNING_CAUSES = frozenset(
    name for name, cls in vars(_exceptions).items()
    if isinstance(cls, type)
    and issubclass(cls, _exceptions.UACPYWarning)
    and cls is not _exceptions.UACPYWarning)

#: The functions that warn with a class handed to them: their own
#: ``warnings.warn`` passes it on, and every call of the helper is held to
#: the rule instead (``_plot_warn``'s second argument, ``give_notice``'s
#: third, a ``Notice``'s ``category``). ``_reemit_job_warnings`` re-raises a
#: worker's warning under the category the worker recorded, and
#: ``_announce_engine_settings`` a notice under the class it carries.
_WARNING_FORWARDERS = {
    ('visualization/plots/_common.py', '_plot_warn'),
    ('models/_notices.py', 'give_notice'),
    ('models/_notices.py', 'message_notice'),
    ('core/run_settings.py', '_saved_notice'),
    ('parallel.py', '_reemit_job_warnings'),
    ('data/_provenance_notice.py', 'fetch_with_one_provenance_notice'),
    ('models/base.py', '_announce_engine_settings'),
}


#: The readers of a saved settings record: a notice saved before notices
#: carried their class, or under a name that is not a uacpy warning class,
#: has no cause to restate and is read as the base class.
_SAVED_NOTICE_READERS = {
    ('core/run_settings.py', '_saved_notice'),
    ('core/run_settings.py', '_with_saved_notes'),
}


def _warning_categories(tree):
    """``(lineno, enclosing function, category source)`` for every
    ``warnings.warn`` call, every ``_plot_warn`` / ``give_notice`` call in
    ``tree``, and every ``Notice`` record; the category source is ``None``
    when the call names none."""
    found = []

    def category_of(call, position, keyword):
        if len(call.args) > position:
            return ast.unparse(call.args[position])
        for kw in call.keywords:
            if kw.arg == keyword:
                return ast.unparse(kw.value)
        return None

    def walk(node, fn):
        for ch in ast.iter_child_nodes(node):
            inner = (ch.name if isinstance(ch, (ast.FunctionDef,
                                                ast.AsyncFunctionDef))
                     else fn)
            if isinstance(ch, ast.Call):
                f = ch.func
                if (isinstance(f, ast.Attribute) and f.attr == 'warn'
                        and isinstance(f.value, ast.Name)
                        and f.value.id in ('warnings', '_warnings')):
                    found.append((ch.lineno, fn, category_of(ch, 1, 'category')))
                elif isinstance(f, ast.Name) and f.id == '_plot_warn':
                    found.append((ch.lineno, fn, category_of(ch, 1, 'category')))
                elif isinstance(f, ast.Name) and f.id == 'give_notice':
                    found.append((ch.lineno, fn, category_of(ch, 2, 'category')))
                elif isinstance(f, ast.Name) and f.id == 'Notice':
                    found.append((ch.lineno, fn, category_of(ch, 2, 'category')))
            walk(ch, inner)
    walk(tree, None)
    return found


def test_every_warning_names_its_cause():
    """Every warning package code emits passes one of the
    :class:`~uacpy.core.exceptions.UACPYWarning` causes, so a user can filter
    by cause and a test can assert one; a bare ``UserWarning`` (or a call
    that names no category, which means ``UserWarning``) is refused. The
    forwarding helpers pass their caller's class on and are held to the rule
    at their calls."""
    offenders, checked = [], 0
    for path in sorted(_PACKAGE.rglob('*.py')):
        rel = path.relative_to(_PACKAGE).as_posix()
        if rel.split('/')[0] in ('tests', 'examples', 'third_party'):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for lineno, fn, category in _warning_categories(tree):
            checked += 1
            if category in _WARNING_CAUSES:
                continue
            if (rel, fn) in _WARNING_FORWARDERS and category == 'category':
                continue
            # a sink forwarding a budget Notice's own class
            if category == 'notice.category':
                continue
            # a notice read back from a record saved without its class
            if (rel, fn) in _SAVED_NOTICE_READERS and category == 'UACPYWarning':
                continue
            offenders.append(f'{rel}:{lineno} ({fn}): {category}')
    # A sweep that stopped recognising the calls would pass silently.
    assert checked >= 250, f'only {checked} warning calls found'
    assert not offenders, (
        f'warnings that name no uacpy cause (pass one of '
        f'{", ".join(sorted(_WARNING_CAUSES))}):\n  '
        + '\n  '.join(offenders))


# ── layering: which layer may import which ────────────────────────────────

#: The layers, bottom first. A module's layer is its subpackage
#: (``uacpy.<layer>``, with ``core.results`` and ``core.acoustics`` apart
#: from ``core``) or, for a top-level module, its own name; ``__root__`` is
#: ``uacpy/__init__.py``. A layer imports only from its own rank or below;
#: the exceptions are listed in :data:`_UPWARD_IMPORTS`.
_LAYER_ORDER = (
    ('core', 'core.acoustics', '_log', '_stack', '_version'),
    ('acoustic_signal',),
    ('comms', 'noise'),
    ('core.results',),
    ('sonar', 'analytic'),
    ('io', 'data'),
    ('models',),
    ('parallel',),
    ('visualization',),
    ('__root__', 'acoustics', 'materials', 'metrics', 'plot', 'units'),
)

#: ``(importer, kind, imported module)`` for every import that points up the
#: layer order. The list may only shrink: a new upward import fails
#: :func:`test_layering`, and so does an entry whose import is gone.
_UPWARD_IMPORTS = frozenset({
    ('uacpy.acoustic_signal.beamforming', 'lazy', 'uacpy.core.results'),
    ('uacpy.core._plotting', 'lazy', 'uacpy.visualization.plots'),
    ('uacpy.core.metrics', 'top', 'uacpy.core.results'),
})

#: ``(importer, module, name)`` for every module-private name (``_x``, not
#: ``__x__``) imported across a layer boundary. A ``_``-prefixed module may
#: cross a package boundary only when ``uacpy.tests._internal_modules``
#: declares it (``test_packaging`` gates that), and the names such a module
#: is imported for are covered by its declaration, so they are not repeated
#: here; a ``_name`` inside any other module belongs to that module. The list
#: may only shrink, as :data:`_UPWARD_IMPORTS` does.
_PRIVATE_NAMES_ACROSS_LAYERS = frozenset({
    # The labelled form of a user-level function: the public one carries no
    # ``who``, and a method (or engine) that delegates passes its own name so
    # a refusal names the call the user made.
    ('uacpy.core.results.field', 'uacpy.acoustic_signal.channel',
     '_broadband_propagation_loss'),
    ('uacpy.core.results.field', 'uacpy.acoustic_signal.channel',
     '_gate_transfer_function'),
    ('uacpy.core.results.field', 'uacpy.acoustic_signal.channel',
     '_transfer_function_from_impulse_response'),
    ('uacpy.core.results.field', 'uacpy.acoustic_signal.spectrum_at',
     '_tone_phasor'),
    ('uacpy.core.results.modes', 'uacpy.core.acoustics.modal',
     '_modal_attenuation'),
    ('uacpy.core.results.modes', 'uacpy.core.acoustics.modal',
     '_modal_grazing_angles'),
    ('uacpy.core.results.rays', 'uacpy.acoustic_signal.channel',
     '_arrival_grid_transfer_function'),
    ('uacpy.core.results.rays', 'uacpy.acoustic_signal.channel',
     '_arrival_transfer_function'),
    ('uacpy.core.results.rays', 'uacpy.acoustic_signal.channel',
     '_simulate_arrival_grid'),
    ('uacpy.core.results.rays', 'uacpy.acoustic_signal.delay_profile',
     '_channel_regime'),
    ('uacpy.core.results.rays', 'uacpy.acoustic_signal.delay_profile',
     '_coherence_bandwidth'),
    ('uacpy.core.results.rays', 'uacpy.acoustic_signal.delay_profile',
     '_energy_support'),
    ('uacpy.core.results.rays', 'uacpy.acoustic_signal.delay_profile',
     '_rms_delay_spread'),
    ('uacpy.core.results.rays', 'uacpy.acoustic_signal.delay_profile',
     '_synthesis_band'),
    ('uacpy.core.results.rays', 'uacpy.comms.channel',
     '_pulse_shaped_taps'),
    ('uacpy.models.bellhop._synthesis', 'uacpy.acoustic_signal.delay_profile',
     '_fold_notice'),
    ('uacpy.models.sparc._extract', 'uacpy.acoustic_signal.spectrum_at',
     '_tone_phasor'),
    ('uacpy.visualization.plots.rays_modes', 'uacpy.acoustic_signal.delay_profile',
     '_energy_support'),
    ('uacpy.core.results.modes', 'uacpy.core.acoustics.modal',
     '_mode_shapes_at'),
    ('uacpy.core.results.greens_function', 'uacpy.core.acoustics.wavenumber',
     '_warn_zero_ranges'),
    # The one weighted sum superpose forms; the n = 1 weight a run applies
    # calls it to refuse a bad weight in the caller's own terms
    # (``<model>: Source(weights=…)``).
    ('uacpy.models._stacking', 'uacpy.core.results.stack',
     '_weighted_slab_sum'),
    ('uacpy.models._stacking', 'uacpy.core.results.stack',
     '_check_stack_weightable'),
    # The dB arm of the same rule, raised in stage 3 before the engine runs.
    ('uacpy.models._stacking', 'uacpy.core.results.stack',
     '_dB_weight_refusal'),
    ('uacpy.models.bounce._model', 'uacpy.io.refl_io', '_scale_irc_impedance'),
    ('uacpy.models.oases._common', 'uacpy.io.oases_writer',
     '_OASES_MAX_WAVENUMBERS'),
    ('uacpy.models.oases._sampling', 'uacpy.io.oases_writer',
     '_OASES_MAX_WAVENUMBERS'),
    ('uacpy.models.oases.oasn', 'uacpy.io.oases_writer',
     '_resolve_freq_sweep'),
    ('uacpy.models.oases.oasn', 'uacpy.io.oases_writer',
     '_check_oasn_discrete_sources'),
    ('uacpy.models.oases.oasn', 'uacpy.io.oases_writer',
     '_check_oasn_noise_level'),
    ('uacpy.models.oases.oasn', 'uacpy.io.oases_writer',
     '_check_oasn_replica_counts'),
    ('uacpy.models.oases.oass', 'uacpy.io.oases_writer',
     '_OASES_MAX_WAVENUMBERS'),
    ('uacpy.models.oases.oass', 'uacpy.io.oases_writer',
     '_OASS_REVERB_OPTIONS'),
    ('uacpy.models.oases.oass', 'uacpy.io.oases_writer',
     '_check_oass_range_count'),
    ('uacpy.models.oases.oast', 'uacpy.io.oases_writer',
     '_resolve_freq_sweep'),
})

#: Layers ``core`` may not import when one of its modules is imported.
_ABOVE_CORE_AT_IMPORT = ('models', 'io', 'visualization', 'data')


def _layer(module):
    parts = module.split('.')
    if len(parts) == 1:
        return '__root__'
    if parts[1] == 'core' and len(parts) > 2 and parts[2] in ('results',
                                                              'acoustics'):
        return f'core.{parts[2]}'
    return parts[1]


def _import_graph():
    """``(module, kind, target, name)`` for every uacpy import in the
    package; ``target`` is always a module."""
    return [(module, kind, target, name)
            for module, tree in _package_modules().items()
            for kind, target, name, _bound, _line in _uacpy_imports(module,
                                                                    tree)]


def _layering_violations(edges):
    """What :func:`test_layering` reports for ``edges``, as a list of
    sentences; empty when the graph keeps every rule."""
    rank = {layer: r for r, layers in enumerate(_LAYER_ORDER)
            for layer in layers}
    modules = {module for module, *_ in edges}
    unranked = sorted({_layer(m) for m in modules} - set(rank))
    if unranked:
        return [f'layer(s) {unranked} have no place in _LAYER_ORDER']
    problems = []

    graph = collections.defaultdict(set)
    for module, kind, target, _name in edges:
        if kind == 'top' and target != module:
            graph[module].add(target)
    cycle = _a_cycle(graph)
    if cycle:
        problems.append('module-level import cycle: ' + ' -> '.join(cycle))

    for module, kind, target, _name in edges:
        if (kind == 'top' and _layer(module).split('.')[0] == 'core'
                and _layer(target) in _ABOVE_CORE_AT_IMPORT):
            problems.append(f'{module} imports {target} at module level')

    upward = {(module, kind, target) for module, kind, target, _name in edges
              if kind != 'tc' and rank[_layer(module)] < rank[_layer(target)]}
    private = {(module, target, name) for module, _kind, target, name in edges
               if name and name.startswith('_') and not name.startswith('__')
               and _layer(module) != _layer(target)
               and private_module(target) not in PACKAGE_INTERNAL_MODULES}
    for label, measured, allowed in (
            ('upward import', upward, _UPWARD_IMPORTS),
            ('private name across layers', private,
             _PRIVATE_NAMES_ACROSS_LAYERS)):
        problems += [f'new {label}: {entry}'
                     for entry in sorted(measured - allowed)]
        problems += [f'{label} gone, delete its allow-list entry: {entry}'
                     for entry in sorted(allowed - measured)]
    return problems


def _a_cycle(graph):
    """One cycle of ``graph`` as a list of nodes (first repeated last), or
    ``None``."""
    state, stack = {}, []

    def walk(node):
        state[node] = 'open'
        stack.append(node)
        for nxt in sorted(graph.get(node, ())):
            if state.get(nxt) == 'open':
                return stack[stack.index(nxt):] + [nxt]
            if nxt not in state:
                found = walk(nxt)
                if found:
                    return found
        stack.pop()
        state[node] = 'done'
        return None

    for node in sorted(graph):
        if node not in state:
            found = walk(node)
            if found:
                return found
    return None


def test_layering():
    """The package imports downward: no module-level import cycle; ``core``
    loads no ``models``, ``io``, ``visualization`` or ``data`` module when
    it is imported; and the imports that point up :data:`_LAYER_ORDER`, and
    the module-private names that cross a layer, are exactly the listed
    ones. Both lists only shrink: a new entry is a new dependency to route
    downward, and a stale one is a fix to record by deleting it."""
    edges = _import_graph()
    # A sweep that stopped resolving imports would find no violation.
    assert len(edges) >= 1500, f'only {len(edges)} imports were resolved'
    problems = _layering_violations(edges)
    assert not problems, '\n'.join(problems)


def test_every_core_module_lists_its_public_definitions_in_all():
    """One ``__all__`` rule for the core leaf modules: a module whose name is
    public lists every public name it defines (a function, a class, a
    module-level constant). A listed name it does not define is one it
    re-exports on purpose (``environment`` restates the carriers)."""
    core = _PACKAGE / 'core'
    problems = []
    for path in sorted([*core.glob('*.py'), *(core / 'acoustics').glob('*.py')]):
        if path.name.startswith('_'):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        defined, listed = [], None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                target = (node.targets[0] if isinstance(node, ast.Assign)
                          else node.target)
                name = getattr(target, 'id', '_')
                if name == '__all__':
                    listed = ast.literal_eval(node.value)
                    continue
            else:
                continue
            if not name.startswith('_'):
                defined.append(name)
        rel = path.relative_to(_PACKAGE)
        if listed is None:
            problems.append(f'{rel}: no __all__')
        elif set(defined) - set(listed):
            problems.append(f'{rel}: __all__ omits '
                            f'{sorted(set(defined) - set(listed))}')
    assert not problems, '\n'.join(problems)


def test_the_signal_layer_imports_without_the_results_layer():
    """``acoustic_signal`` sits below ``core.results`` in
    :data:`_LAYER_ORDER`, so importing it in a fresh interpreter must not
    load ``uacpy.core.results``."""
    import subprocess
    import sys
    probe = ("import sys, uacpy, uacpy.acoustic_signal.spectral, "
             "uacpy.acoustic_signal.timefreq, uacpy.acoustic_signal.channel, "
             "uacpy.acoustic_signal.beamforming; "
             "print(uacpy.__file__); "
             "print('uacpy.core.results' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', probe], check=True,
                         capture_output=True, text=True,
                         cwd=_PACKAGE.parent).stdout.split('\n')
    assert Path(out[0]).resolve().parent == _PACKAGE, out[0]
    assert out[1] == 'False'


# A pin on the exception TYPE alone passes for any refusal of that type: a
# test built to reach refusal B meets refusal A first and stays green, and a
# method called with a signature it does not take raises the TypeError a
# type-only pin accepts. Each entry is ``'file::qualname': reason``, for a test
# whose exception carries no stable message to name.
_TYPE_ONLY_RAISES_ALLOWED = {
}


def _package_error_names():
    """Names of the exception classes the package defines."""
    return {node.name for _, tree in _sources() for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and node.name.endswith(('Error', 'Exception'))}


_BUILTIN_ERRORS = frozenset(
    name for name, obj in vars(builtins).items()
    if isinstance(obj, type) and issubclass(obj, Exception)
    and not issubclass(obj, Warning))


def _raised_names(node):
    if isinstance(node, ast.Tuple):
        return [name for elt in node.elts for name in _raised_names(elt)]
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    return []


def _type_only_raises(tree, error_names):
    """``(qualname, line)`` of each ``pytest.raises`` on a package or builtin
    error that carries no ``match=``. A class the test module defines itself
    (a sentinel a stub raises) is the test's own and needs no message."""
    local = {node.name for node in ast.walk(tree)
             if isinstance(node, ast.ClassDef)}
    found = []

    def visit(node, qual):
        for child in ast.iter_child_nodes(node):
            name = getattr(child, 'name', None)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                visit(child, f'{qual}.{name}' if qual else name)
                continue
            if (isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == 'raises'
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == 'pytest'
                    and child.args
                    and not any(k.arg == 'match' for k in child.keywords)
                    and any((n in error_names or n in _BUILTIN_ERRORS)
                            and n not in local
                            for n in _raised_names(child.args[0]))):
                found.append((qual, child.lineno))
            visit(child, qual)

    visit(tree, '')
    return found


def test_every_refusal_pin_names_its_refusal():
    """Every ``pytest.raises`` on a uacpy or builtin error in the suite
    carries ``match=``, except the listed tests, and every listed test still
    has the type-only pin its entry excuses."""
    error_names = _package_error_names()
    offenders, allowed_seen = [], set()
    for path in sorted((_PACKAGE / 'tests').glob('test_*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for qual, line in _type_only_raises(tree, error_names):
            key = f'{path.name}::{qual}'
            if key in _TYPE_ONLY_RAISES_ALLOWED:
                allowed_seen.add(key)
            else:
                offenders.append(f'{path.name}:{line} ({qual})')
    assert not offenders, (
        'pytest.raises on a uacpy or builtin error without match= passes for '
        'any refusal of that type; add match= naming the knob or value the '
        'refusal is about:\n  ' + '\n  '.join(offenders))
    assert allowed_seen == set(_TYPE_ONLY_RAISES_ALLOWED), (
        'allow-list entries with no type-only pin left: '
        f'{sorted(set(_TYPE_ONLY_RAISES_ALLOWED) - allowed_seen)}')


def test_the_refusal_pin_gate_sees_each_spelling():
    """The gate reads the call forms the suite uses: a bare name, a dotted
    name, a tuple; and leaves a pinned call and the test's own sentinel alone."""
    tree = ast.parse(
        'import pytest, uacpy\n'
        'class _Stop(Exception):\n'
        '    pass\n'
        'def test_a():\n'
        '    with pytest.raises(ConfigurationError):\n'
        '        pass\n'
        '    with pytest.raises(uacpy.ConfigurationError) as info:\n'
        '        pass\n'
        '    with pytest.raises((TypeError, ValueError)):\n'
        '        pass\n'
        '    with pytest.raises(ConfigurationError, match="x"):\n'
        '        pass\n'
        '    with pytest.raises(_Stop):\n'
        '        pass\n')
    assert _type_only_raises(tree, {'ConfigurationError'}) == [
        ('test_a', 5), ('test_a', 7), ('test_a', 9)]


def _public_named_surfaces():
    """Every name a caller of ``uacpy``, ``uacpy.acoustic_signal``,
    ``uacpy.comms`` or ``uacpy.sonar`` types: each export's call parameters
    (``mod.callable(param)``) and each exported class's public attributes
    (``mod.Class.attr``).

    Both halves, because a unit suffix has to hold wherever the name is
    read: ``AmbiguityResult`` is a NamedTuple, so ``delays_s`` is a keyword
    at construction *and* an attribute afterwards. Sweeping the surfaces
    together is what makes a suffix rule checkable everywhere at once rather
    than at the one site an audit happened to open."""
    import inspect
    import uacpy.acoustic_signal
    import uacpy.comms
    import uacpy.sonar

    sites = set()
    for module in (uacpy, uacpy.acoustic_signal, uacpy.comms, uacpy.sonar):
        for name in getattr(module, '__all__', ()):
            obj = getattr(module, name, None)
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            try:
                signature = inspect.signature(obj)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                for param in signature.parameters:
                    sites.add(f"{module.__name__}.{name}({param})")
            if inspect.isclass(obj):
                for attr in vars(obj):
                    if not attr.startswith('_'):
                        sites.add(f"{module.__name__}.{name}.{attr}")
                # A namedtuple subclassed to carry methods keeps its field
                # descriptors on the base it was built from, so ``vars`` on
                # the subclass alone stops seeing them — the fields are still
                # readable attributes, and the sweep went quiet about them
                # rather than reporting fewer. Read them off ``_fields``,
                # which does not care which class in the chain holds them.
                for attr in getattr(obj, '_fields', ()):
                    if not attr.startswith('_'):
                        sites.add(f"{module.__name__}.{name}.{attr}")
    return sites


def _sites_with_suffix(suffix):
    """The public surfaces whose *name* ends in ``suffix`` — the trailing
    ``)`` of a parameter site is stripped before matching."""
    return sorted(s for s in _public_named_surfaces()
                  if s.rstrip(')').endswith(suffix))


class TestTheSecondsSuffixIsNotSpentOnMetresPerSecond:
    """``_s`` is this package's suffix for seconds — eleven public sites use
    it that way (``delays_s``, ``pulse_length_s``, ``integration_time_s``, …)
    — so a public ``_ms`` reads as milliseconds, which is what ``_ms``
    already means at every private site that has one (``t_ms``,
    ``delays_ms``, ``time_ms`` are all milliseconds). Metres per second is
    spelled ``_mps``.
    """

    def test_the_sea_surface_generator_takes_wind_speed_kn(self):
        import inspect
        params = inspect.signature(uacpy.generate_sea_surface).parameters
        assert 'wind_speed_kn' in params
        # the default is 10 m/s, written in the knots the argument takes
        from uacpy.core.altimetry import DEFAULT_SEA_SURFACE_WIND_KN
        assert params['wind_speed_kn'].default == DEFAULT_SEA_SURFACE_WIND_KN
        assert float(uacpy.units.knots_to_ms(DEFAULT_SEA_SURFACE_WIND_KN)) == pytest.approx(10.0, rel=1e-15)

    def test_the_wind_speed_kn_keyword_sets_the_wave_height(self):
        # The name is pinned on the live signature above; this pins that it
        # is the wind speed, so the gate cannot be satisfied by a parameter
        # that merely spells itself right.
        calm = uacpy.generate_sea_surface(
            2000.0, wind_speed_kn=10.0, n_points=256,
            rng=np.random.default_rng(3))
        blow = uacpy.generate_sea_surface(
            2000.0, wind_speed_kn=30.0, n_points=256,
            rng=np.random.default_rng(3))
        # Pierson-Moskowitz: Hs = 0.021*U^2, so 30 kn is 9x the 10 kn sea.
        assert float(np.std(blow[:, 1])) > 5.0 * float(np.std(calm[:, 1]))

    def test_the_seconds_spelling_of_the_wind_speed_keyword_is_not_accepted(self):
        # The other side of the same boundary: one spelling reaches the
        # generator and the other is a TypeError, so the two cannot both be
        # live at once.
        with pytest.raises(TypeError, match='wind_speed_ms'):
            uacpy.generate_sea_surface(2000.0, wind_speed_ms=5.0, n_points=64)

    def test_the_seconds_suffix_is_already_spoken_for_across_the_package(self):
        """The premise the ``_mps`` spelling rests on, measured rather than
        asserted: ``_s`` names seconds on every public surface that carries
        it — call parameters and readable attributes alike, which is why the
        sweep covers attributes too. ``AmbiguityResult.delays_s`` and the
        ``ChannelTaps`` pair (``delays_s``, ``first_arrival_s``) are both
        passable and readable, so each counts twice. If this count moves
        the convention has changed and the rule below has to be restated.

        The count is a tripwire, not a budget: every site it admits must
        mean seconds. It last moved when the generic channel functions
        were exported — the power-delay-profile group's four ``delays_s``,
        the two ``ChannelRegime`` fields that became public with the type,
        ``arrival_transfer_function``'s ``delays_s`` / ``delays_imag_s``,
        ``pulse_shaped_taps``'s ``delays_s``, and the same pair on
        ``simulate_arrival_reception``; then by three when the arrival
        computations of the result types became functions:
        ``synthesis_band``'s and ``fold_notice``'s ``delays_s`` and
        ``received_amplitudes``'s ``delays_imag_s``; then down by one when
        ``fold_notice``, which takes its caller's error text, went private."""
        seconds = [s for s in _sites_with_suffix('_s')
                   if not s.rstrip(')').endswith('_ms')]
        assert len(seconds) == 28, seconds
        for site in ('uacpy.acoustic_signal.AmbiguityResult.delays_s',
                     'uacpy.comms.ChannelTaps(delays_s)',
                     'uacpy.comms.ChannelTaps.delays_s',
                     'uacpy.comms.ChannelTaps(first_arrival_s)',
                     'uacpy.comms.ChannelTaps.first_arrival_s',
                     'uacpy.acoustic_signal.rms_delay_spread(delays_s)',
                     'uacpy.acoustic_signal.ChannelRegime(symbol_duration_s)',
                     'uacpy.acoustic_signal.arrival_transfer_function(delays_imag_s)',
                     'uacpy.comms.pulse_shaped_taps(delays_s)',
                     'uacpy.acoustic_signal.simulate_arrival_reception(delays_imag_s)'):
            assert site in seconds, site

    def test_no_public_surface_spells_metres_per_second_as_ms(self):
        offenders = _sites_with_suffix('_ms')
        assert not offenders, (
            "public name(s) ending in `_ms`, which reads as milliseconds "
            "next to the `_s`-for-seconds sites:\n" + "\n".join(offenders))

    def test_the_mps_spelling_is_the_one_the_sweep_finds(self):
        """The far side of the previous gate: silence there must mean the
        sweep looked and found nothing, not that it sees no speed at all.

        ``uacpy.comms.doppler_from_speed`` spells its platform speed
        ``_mps``; the wind arguments are knots (``_kn``, pinned below); a
        sound speed is ``sound_speed``, whose unit the package fixes as m/s."""
        assert _sites_with_suffix('_mps') == [
            'uacpy.comms.doppler_from_speed(speed_mps)',
        ]


class TestTheWindArgumentsTakeKnots:
    """Every public wind argument is ``wind_speed_kn`` and its value is in
    knots; a law stated in m/s converts once, at its entry. Each m/s-native
    entry is pinned against its own m/s formula fed the converted value, so
    dropping a conversion (or adding a second one) fails here."""

    _ENTRIES = ('uacpy.noise.WenzNoise', 'uacpy.noise.wind_noise_level',
                'uacpy.generate_sea_surface',
                'uacpy.core.altimetry.sea_surface_n_points',
                'uacpy.core.altimetry.Altimetry.from_sea_state',
                'uacpy.sonar.chapman_harris_surface',
                'uacpy.sonar.apl_uw_surface_backscatter',
                'uacpy.core.acoustics.bubble_surface_loss')

    @staticmethod
    def _resolve(dotted):
        import importlib
        parts = dotted.split('.')
        for cut in range(len(parts), 0, -1):
            try:
                obj = importlib.import_module('.'.join(parts[:cut]))
            except ImportError:
                continue
            for name in parts[cut:]:
                obj = getattr(obj, name)
            return obj
        raise AssertionError(dotted)

    def test_every_wind_entry_names_its_argument_in_knots(self):
        import inspect
        for dotted in self._ENTRIES:
            params = inspect.signature(self._resolve(dotted)).parameters
            assert 'wind_speed_kn' in params, dotted
            assert not [p for p in params if p.endswith('_mps')], dotted

    def test_the_suffix_sweep_finds_the_knots_sites(self):
        assert _sites_with_suffix('_kn') == [
            'uacpy.generate_sea_surface(wind_speed_kn)',
            'uacpy.sonar.apl_uw_surface_backscatter(wind_speed_kn)',
            'uacpy.sonar.chapman_harris_surface(wind_speed_kn)',
        ]

    def test_the_coates_wind_term_reads_its_m_per_s_from_knots(self):
        from uacpy.noise import ambient
        f = np.array([1000.0, 10000.0])
        got = ambient._wind_coates(f, wind_speed_kn=uacpy.units.ms_to_knots(10.0))
        fk = f / 1000.0
        want = (50.0 + 7.5 * np.sqrt(10.0) + 20.0 * np.log10(fk)
                - 40.0 * np.log10(fk + 0.4))
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_the_merklinger_wind_term_reads_knots_as_given(self):
        """DRDC eq. 8 is fitted in knots and reads ``u`` as given: 59.66 dB
        re 1 uPa^2/Hz at 1 kHz for 10 kn (the knots-native level the
        WenzNoise doctest's 59.7 dB total rounds); a conversion on the way in
        would read 19.4 kn or 5.1 kn instead."""
        from uacpy.noise import wind_noise_level
        level = wind_noise_level(np.array([1000.0]), wind_speed_kn=10.0)[0]
        assert level == pytest.approx(59.66190058726154, abs=1e-9)

    def test_the_apl_surface_model_reads_its_m_per_s_from_knots(self):
        from uacpy.sonar import scattering
        theta = np.array([10.0, 30.0, 60.0])
        got = scattering.apl_uw_surface_backscatter(
            frequency=25e3, grazing_deg=theta,
            wind_speed_kn=uacpy.units.ms_to_knots(8.0))
        want = scattering._strength(theta, 25e3, 8.0)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_the_sea_surface_sizing_reads_its_m_per_s_from_knots(self):
        from uacpy.core.altimetry import sea_surface_n_points
        from uacpy.core.constants import STANDARD_GRAVITY_M_S2
        lambda_p = 2 * np.pi * 10.0 ** 2 / STANDARD_GRAVITY_M_S2     # U = 10 m/s
        n, needed = sea_surface_n_points(10_000.0, uacpy.units.ms_to_knots(10.0))
        assert needed == int(np.ceil(8 * 10_000.0 / lambda_p)) + 1


class TestTheHostHasOneAvailableMemoryReader:
    """``uacpy.core._host.available_memory_bytes`` is the one reader the
    memory budget (``models/_budget.py``) weighs every engine's estimate
    against: MemAvailable first (in bytes),
    the sysconf pair without ``/proc/meminfo``, and ``None`` — not a negative
    size — when sysconf reports -1."""

    @staticmethod
    def _procfs(monkeypatch, text=None, pages=10):
        import builtins
        import io
        import os
        real_open = builtins.open

        def fake_open(path, *args, **kwargs):
            if path == '/proc/meminfo':
                if text is None:
                    raise OSError('no procfs')
                return io.StringIO(text)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, 'open', fake_open)
        monkeypatch.setattr(os, 'sysconf', lambda name: (
            pages if name == 'SC_AVPHYS_PAGES' else 4096))
        from uacpy.core import _host
        monkeypatch.setattr(_host, '_cgroup_memory_headroom', lambda: None)

    def test_memavailable_is_read_in_bytes(self, monkeypatch):
        from uacpy.core._host import available_memory_bytes
        self._procfs(monkeypatch, "MemTotal:  100 kB\nMemFree:  1 kB\n"
                                  "MemAvailable:  12345 kB\n")
        assert available_memory_bytes() == 12345 * 1024

    def test_without_the_file_the_sysconf_pair_is_used(self, monkeypatch):
        from uacpy.core._host import available_memory_bytes
        self._procfs(monkeypatch, None, pages=10)
        assert available_memory_bytes() == 40960

    def test_an_unknown_page_count_is_none(self, monkeypatch):
        from uacpy.core._host import available_memory_bytes
        self._procfs(monkeypatch, None, pages=-1)
        assert available_memory_bytes() is None

    @staticmethod
    def _cgroup(tmp_path, lines, files):
        """A fake ``/proc/self/cgroup`` + cgroup tree under ``tmp_path``."""
        proc = tmp_path / 'cgroup'
        proc.write_text('\n'.join(lines) + '\n')
        root = tmp_path / 'fs'
        for rel, text in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return str(proc), str(root)

    def test_a_v2_cgroup_limit_caps_the_host_figure(self, tmp_path,
                                                    monkeypatch):
        """RA-CONTRACT-14: inside a container the host's MemAvailable
        overstates what the job may allocate; the cgroup's
        ``memory.max - memory.current`` wins when it is smaller, and a
        parent's limit binds the child."""
        from uacpy.core import _host
        proc, root = self._cgroup(tmp_path, ['0::/job/step'], {
            'job/memory.max': '1000000', 'job/memory.current': '400000',
            'job/step/memory.max': 'max', 'job/step/memory.current': '300000',
        })
        assert _host._cgroup_memory_headroom(proc, root) == 600000
        monkeypatch.setattr(_host, '_cgroup_memory_headroom', lambda: 600000)
        monkeypatch.setattr(_host, '_host_available_memory_bytes',
                            lambda: 10 ** 12)
        assert _host.available_memory_bytes() == 600000

    def test_an_unlimited_or_absent_cgroup_leaves_the_host_figure(self,
                                                                 tmp_path):
        from uacpy.core import _host
        proc, root = self._cgroup(tmp_path, ['0::/job'], {
            'job/memory.max': 'max', 'job/memory.current': '5'})
        assert _host._cgroup_memory_headroom(proc, root) is None
        assert _host._cgroup_memory_headroom(
            str(tmp_path / 'missing'), root) is None

    def test_a_v1_memory_controller_limit_is_read(self, tmp_path):
        from uacpy.core import _host
        proc, root = self._cgroup(
            tmp_path, ['4:cpu,cpuacct:/x', '3:memory:/job'], {
                'memory/job/memory.limit_in_bytes': '2048',
                'memory/job/memory.usage_in_bytes': '48'})
        assert _host._cgroup_memory_headroom(proc, root) == 2000
        proc, root = self._cgroup(tmp_path, ['3:memory:/job'], {
            'memory/job/memory.limit_in_bytes': str(1 << 62),
            'memory/job/memory.usage_in_bytes': '48'})
        assert _host._cgroup_memory_headroom(proc, root) is None

    def test_every_engine_weighs_memory_through_the_budget(self):
        from uacpy.core import _host
        from uacpy.models import _budget
        import uacpy.io.grn_reader as grn
        import uacpy.models.bellhop._plan as bellhop
        import uacpy.models.kraken._grid as kraken
        import uacpy.models.ram.mpirams as ram
        import uacpy.models.scooter._plan as scooter
        import uacpy.models.sparc._plan as sparc
        assert _budget.available_memory_bytes is _host.available_memory_bytes
        for mod in (bellhop, kraken, ram, scooter, sparc):
            assert mod.memory_budget is _budget.memory_budget
            assert not hasattr(mod, 'available_memory_bytes')
        assert not hasattr(ram, '_available_memory_bytes')
        assert not hasattr(grn, 'available_memory_bytes')


class TestACallCarriesWhatItAskedFor:
    """What one call asked for travels on the checked call (``_RunCall``),
    handed to the stages that read it, never through context variables: the
    one ContextVar of the models is ``_INSIDE_RUN``, which tells a run
    started inside another that it is not the call the user made."""

    def test_the_models_declare_one_context_variable(self):
        found = []
        for path in sorted((_PACKAGE / 'models').rglob('*.py')):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                func = getattr(value, 'func', None)
                name = (func.attr if isinstance(func, ast.Attribute)
                        else getattr(func, 'id', None))
                if name == 'ContextVar':
                    targets = (node.targets if isinstance(node, ast.Assign)
                               else [node.target])
                    found += [(path.name, t.id) for t in targets]
        assert found == [('_notices.py', '_INSIDE_RUN')]

    def test_the_producer_engines_run_the_base_projection(self):
        """``PropagationModel._producer_settings`` projects with the pure
        ``_environment_projection``, which is what a producer's stage 1 is
        only while no producer engine overrides ``_project_environment``."""
        from uacpy.models import OASP, OAST, Bounce
        from uacpy.models.base import PropagationModel
        for cls in (Bounce, OASP, OAST):
            assert (cls._project_environment
                    is PropagationModel._project_environment), cls


class TestEveryLaunchPassesOneRunner:
    """Every engine binary is launched as a ``Launch`` record through
    ``_launch.run_launch``, which reaches the model's ``_run_subprocess``
    (the seam a test or a timeout stops a run at) only through
    ``PropagationModel._launch_binary``."""

    def test_the_subprocess_seam_is_reached_from_one_place(self):
        sites = []
        for path in sorted((_PACKAGE / 'models').rglob('*.py')):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute)
                        and node.attr == '_run_subprocess'):
                    sites.append(path.name)
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == 'run_subprocess'):
                    sites.append(path.name)
        assert sites == ['base.py', 'base.py'], sites


# ── a writer names a knob as its model does ──────────────────────────────

#: The io writers that write a deck for one engine, by module and function.
_WRITER_ENGINES = {
    ('io/mpirams_writer.py', 'write_inpe'): 'RAM',
    ('io/mpirams_writer.py', 'write_sediment_file'): 'RAM',
    ('io/ramsurf_writer.py', 'write_ramin'): 'RAM',
    ('io/oalib_writer.py', 'write_fieldflp'): 'Kraken',
    ('io/oalib_writer.py', 'write_field3dflp'): 'Kraken',
    ('io/oalib_writer.py', 'write_kraken_env_file'): 'Kraken',
    ('io/oalib_writer.py', 'write_scooter_env_file'): 'Scooter',
    ('io/oalib_writer.py', 'write_sparc_env_file'): 'SPARC',
    ('io/oalib_writer.py', 'write_bounce_input_file'): 'Bounce',
    ('io/bellhop_writer.py', 'write_bellhop_env_file'): 'Bellhop',
    ('io/oases_writer.py', 'write_oast_input'): 'OAST',
    ('io/oases_writer.py', 'write_oasn_input'): 'OASN',
    ('io/oases_writer.py', 'write_oasp_input'): 'OASP',
    ('io/oases_writer.py', 'write_oassp_input'): 'OASSP',
    ('io/oases_writer.py', 'write_oasr_input'): 'OASR',
    ('io/oases_writer.py', 'write_oass_input'): 'OASS',
}

#: The deck / Fortran spellings of a knob, by the knob's name (decision 22:
#: a writer takes the knob's name).
_KNOB_ALIASES = {
    'q_factor': ('Q',), 'record_duration': ('T',),
    'dz': ('deltaz',), 'dr': ('deltar',), 'n_pade': ('np_pade', 'np'),
    'n_stability': ('nss', 'ns_stab', 'ns'), 'stability_range_m': ('rs', 'rs_stab'),
    'depth_decimation': ('dzm', 'ndz'), 'c0': ('c0_user',), 'earth_curvature': ('iflat',),
    'n_sediment_points': ('nzs',), 'rams_rotation': ('irot',), 'rams_rotation_angle': ('theta',),
    # FLP MLimit (field.f90:67-68, :185, M = MIN(MLimit, MSrc)) is the n_modes knob
    'n_modes': ('M_limit', 'mode_limit'), 'n_wavenumbers': ('nw_samples', 'nk'),
    'launch_angles': ('alpha',), 'ray_step': ('step',), 'time_max': ('t_max',),
    'n_time_samples': ('n_t_out',), 'march_start': ('t_start',), 'courant_factor': ('t_mult',),
    'freq_min': ('f_min', 'fmin'), 'freq_max': ('f_max', 'fmax'),
    'range_min': ('plot_rmin',), 'range_max': ('plot_rmax',),
    'integrand_plot_step': ('freq_output_increment',),
    'plot_frequency_step': ('freq_output_increment',),
    'plot_angle_step': ('angle_output_increment',),
    'window_sound_speed': ('sound_speed',),
}


def _writer_knob_misnames(sources=None):
    """``[(module:function, parameter, knob)]``: every parameter of a deck
    writer that is a deck spelling of a knob its engine has
    (:data:`_KNOB_ALIASES`) instead of the knob's own name."""
    import inspect
    from uacpy.models._registry import ENGINES
    knobs = {}
    for entry in ENGINES.values():
        cls = entry.load()
        knobs[entry.class_name] = set(inspect.signature(cls.__init__).parameters)
    found = []
    for (rel, func), engine in sorted(_WRITER_ENGINES.items()):
        text = (sources or {}).get(rel)
        if text is None:
            text = (_PACKAGE / rel).read_text(encoding='utf-8')
        tree = ast.parse(text)
        node = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == func)
        a = node.args
        for p in [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]:
            if p in knobs[engine]:
                continue
            for knob, aliases in _KNOB_ALIASES.items():
                if p in aliases and knob in knobs[engine]:
                    found.append((f'{rel}:{func}', p, knob))
    return found


def test_every_deck_writer_names_a_knob_as_its_model_does():
    """Decision 22: a writer parameter that carries a model knob is spelled
    as the knob, so a caller passing ``RAM.dz`` writes ``dz=`` and the deck
    writer's own vocabulary never leaks into the API."""
    bad = _writer_knob_misnames()
    assert not bad, '\n'.join(f'{w}: {p} should be {k}' for w, p, k in bad)


def test_the_writer_knob_gate_sees_a_deck_spelling():
    """The gate reads the writer it is pointed at: ``write_inpe`` spelled
    with the deck's ``deltaz`` is reported."""
    rel = 'io/mpirams_writer.py'
    text = (_PACKAGE / rel).read_text(encoding='utf-8')
    bad = _writer_knob_misnames(
        {rel: text.replace('def write_inpe(\n    filepath: Union[str, Path],\n',
                           'def write_inpe(\n    filepath: Union[str, Path],\n    deltaz,\n', 1)})
    assert ('io/mpirams_writer.py:write_inpe', 'deltaz', 'dz') in bad
