"""The free-name diff that checks a moved function (``_free_names``): it
lists what a function reads from its module and reports every name that
resolves to something else at the destination."""

from pathlib import Path

import pytest

from uacpy.tests._free_names import (
    _ROOT, diff_bindings, free_names, main, resolve)


def _sources(**modules):
    """A ``module_source`` over the synthetic modules ``modules``."""
    table = {name.replace('_', '.'): text for name, text in modules.items()}
    return table.get


class TestFreeNames:

    def test_a_function_reads_module_names_and_builtins_but_not_its_locals(
            self):
        src = ("K = 1\n"
               "def f(a):\n"
               "    b = a + K\n"
               "    return len([helper(x) for x in range(b)])\n")
        assert set(free_names(src, 'f')) == {'K', 'len', 'helper', 'range'}

    def test_a_method_is_named_by_its_dotted_qualname(self):
        src = ("class C:\n"
               "    def m(self):\n"
               "        return _BASE + self.x\n")
        assert set(free_names(src, 'C.m')) == {'_BASE'}

    def test_a_name_read_only_inside_a_nested_lambda_is_free(self):
        src = "def f():\n    return lambda: _HIDDEN\n"
        assert set(free_names(src, 'f')) == {'_HIDDEN'}


class TestResolve:

    def test_an_assignment_resolves_to_its_value(self):
        assert resolve('X', "X = 'a'\n", 'pkg.m') == ("= 'a'",)

    def test_an_import_from_a_tree_module_is_followed_to_its_binding(self):
        src = _sources(pkg_base="X = 'a'\n")
        assert resolve('X', "from pkg.base import X\n", 'pkg.m',
                       src) == ("= 'a'",)

    def test_a_name_with_no_module_binding_is_a_builtin_or_unbound(self):
        assert resolve('len', '', 'pkg.m') == ('builtin len',)
        assert resolve('_nowhere', '', 'pkg.m') == ('unbound',)


class TestDiffBindings:

    OLD = ("_BASE = 'bounce_run'\n"
           "class A:\n"
           "    def write(self):\n"
           "        return _BASE + '.env'\n")

    def test_a_function_moved_next_to_a_same_named_constant_is_reported(
            self):
        new = ("_BASE = 'model'\n"
               "class B:\n"
               "    def write(self):\n"
               "        return _BASE + '.env'\n")
        assert diff_bindings(self.OLD, 'pkg.old', 'A.write',
                             new, 'pkg.new', 'B.write') == {
            '_BASE': (("= 'bounce_run'",), ("= 'model'",))}

    def test_a_function_moved_with_its_constant_reports_nothing(self):
        new = self.OLD.replace('class A', 'class B')
        assert diff_bindings(self.OLD, 'pkg.old', 'A.write',
                             new, 'pkg.new', 'B.write') == {}

    def test_a_destination_that_imports_the_constant_reports_nothing(self):
        new = ("from pkg.old import _BASE\n"
               "def write():\n"
               "    return _BASE + '.env'\n")
        assert diff_bindings(self.OLD, 'pkg.old', 'A.write',
                             new, 'pkg.new', 'write',
                             _sources(pkg_old=self.OLD)) == {}

    def test_a_name_the_destination_does_not_bind_is_reported_unbound(self):
        new = "class B:\n    def write(self):\n        return _BASE\n"
        assert diff_bindings(self.OLD, 'pkg.old', 'A.write',
                             new, 'pkg.new', 'B.write') == {
            '_BASE': (("= 'bounce_run'",), ('unbound',))}


class TestTheTree:

    def test_bounce_write_input_moved_into_scooter_reads_scooter_names(
            self):
        """The move the module docstring describes, on the real files:
        Scooter's ``_BASE_NAME`` takes the place of Bounce's, and Bounce's
        deck writer is not imported there."""
        bounce = (_ROOT / 'uacpy/models/bounce/_model.py').read_text()
        scooter = (_ROOT / 'uacpy/models/scooter/_model.py').read_text()
        differ = diff_bindings(bounce, 'uacpy.models.bounce._model',
                               'Bounce._write_input',
                               scooter, 'uacpy.models.scooter._model',
                               'Scooter._write_input')
        assert differ['_BASE_NAME'] == (("= 'bounce_run'",), ("= 'model'",))
        assert differ['write_bounce_input_file'][1] == ('not read',)

    def test_the_command_line_exits_0_on_a_function_against_itself(
            self, capsys):
        site = f"{_ROOT / 'uacpy/models/bounce/_model.py'}:Bounce._write_input"
        assert main([site, site]) == 0
        assert capsys.readouterr().out == ''

    def test_the_command_line_exits_1_and_names_each_differing_name(
            self, capsys):
        old = f"{_ROOT / 'uacpy/models/bounce/_model.py'}:Bounce._write_input"
        new = f"{_ROOT / 'uacpy/models/scooter/_model.py'}:Scooter._write_input"
        assert main([old, new]) == 1
        assert "_BASE_NAME: = 'bounce_run'  ->  = 'model'" in (
            capsys.readouterr().out)

    def test_an_unknown_qualname_raises(self):
        with pytest.raises(KeyError, match='no scope named'):
            free_names(Path(_ROOT / 'uacpy/models/bounce/_model.py').read_text(),
                       'Bounce._no_such_method')
