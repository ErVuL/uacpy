"""``docs/models/validation.md`` quotes the benchmarks' measured agreement;
this gate ties every number on it to the source that states it.

The page is read in units: a table row, a list item or a paragraph (fenced
code and headings are skipped). A unit names its sources in code spans and
links:

* ``test_benchmarks_<name>.py::test_x`` (or ``::Class::test_x``) — that
  test's source: its marks, docstring and body, since a measurement is stated
  in the docstring and a bound in the assert or the parametrize row;
* ``benchmarks/<table>.txt`` — the comment header of that reference table
  under ``uacpy/tests/data/benchmarks/``;
* a link ``page.md#anchor`` — that section of the linked page.

Every decimal number in a unit (``0.230``; ``+4.35`` is read as ``4.35``;
``1.4e-6``) must appear, as a whole number, in the text of one of the unit's
sources, and a unit that carries a decimal must name a source. Integers
(frequencies, depths, counts) are not checked. Every test in the
``test_benchmarks_*.py`` modules is named somewhere on the page, so a new
benchmark needs its entry.
"""

import ast
import importlib.util
import re
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_REPO = _TESTS.parents[1]
_DOCS = _REPO / 'docs'
PAGE = _DOCS / 'models' / 'validation.md'

_spec = importlib.util.spec_from_file_location('_docs_check_links',
                                               _DOCS / 'check_links.py')
_check_links = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_check_links)
heading_slug = _check_links.heading_slug

_FENCE = re.compile(r'^\s*```')
_HEADING = re.compile(r'^(#{1,6}) (.*)$')
_TEST_ID = re.compile(r'`(test_benchmarks_\w+\.py)((?:::\w+)+)`')
_TABLE_FILE = re.compile(r'`(?:[\w/]*/)?benchmarks/([\w.]+\.txt)`')
_LINK = re.compile(r'\]\(([\w./-]+\.md)#([\w-]+)\)')
_CODE_SPAN = re.compile(r'`[^`]*`')
_LINK_TARGET = re.compile(r'\]\([^)]*\)')
_DECIMAL = re.compile(r'(?<![\w.])(\d+\.\d+(?:e[-+]?\d+)?)(?!\d|\.\d)')


def units(text):
    """``(line number, text)`` of each table row, list item and paragraph
    of ``text`` outside fenced code; headings are not units."""
    found, current, start, in_fence = [], [], 0, False

    def close():
        if current:
            found.append((start, ' '.join(current)))
            current.clear()

    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            close()
            in_fence = not in_fence
            continue
        stripped = line.strip()
        if in_fence or not stripped or _HEADING.match(line):
            close()
            continue
        if stripped.startswith('|') or stripped.startswith('- '):
            close()
            start = number
            current.append(stripped)
            if stripped.startswith('|'):
                close()
            continue
        if not current:
            start = number
        current.append(stripped)
    close()
    return found


def decimals(unit):
    """The decimal numbers ``unit`` states, outside its code spans and link
    targets."""
    prose = _LINK_TARGET.sub(']', _CODE_SPAN.sub(' ', unit))
    return _DECIMAL.findall(prose)


def _test_source(module, qualname):
    """The source of ``module::qualname``, decorators included, or None."""
    path = _TESTS / module
    if not path.is_file():
        return None
    text = path.read_text(encoding='utf-8')
    body = ast.parse(text).body
    node = None
    for name in qualname:
        node = next((n for n in body if isinstance(
            n, (ast.FunctionDef, ast.ClassDef)) and n.name == name), None)
        if node is None:
            return None
        body = node.body
    if not isinstance(node, ast.FunctionDef):
        return None
    first = min([d.lineno for d in node.decorator_list] + [node.lineno])
    return '\n'.join(text.splitlines()[first - 1:node.end_lineno])


def _table_header(name):
    """The ``#`` comment header of reference table ``name``, or None."""
    path = _TESTS / 'data' / 'benchmarks' / name
    if not path.is_file():
        return None
    return '\n'.join(line for line in path.read_text(encoding='utf-8')
                     .splitlines() if line.startswith('#'))


def _section(page, anchor):
    """The text of the section of ``page`` whose heading anchors as
    ``anchor``, or None."""
    path = (PAGE.parent / page).resolve()
    if not path.is_file():
        return None
    lines, out, level = path.read_text(encoding='utf-8').splitlines(), None, 0
    for line in lines:
        match = _HEADING.match(line)
        if match and out is not None and len(match.group(1)) <= level:
            break
        if match and out is None and heading_slug(match.group(2)) == anchor:
            out, level = [], len(match.group(1))
        if out is not None:
            out.append(line)
    return None if out is None else '\n'.join(out)


def sources(unit):
    """``(name, text)`` of every source ``unit`` names; text is None for a
    name that does not resolve."""
    found = []
    for module, path in _TEST_ID.findall(unit):
        found.append((f'{module}{path}',
                      _test_source(module, path.split('::')[1:])))
    for name in _TABLE_FILE.findall(unit):
        found.append((f'benchmarks/{name}', _table_header(name)))
    for page, anchor in _LINK.findall(unit):
        found.append((f'{page}#{anchor}', _section(page, anchor)))
    return found


def _states(text, number):
    return re.search(rf'(?<![\d.]){re.escape(number)}(?!\d|\.\d)',
                     text) is not None


def unbacked_numbers(text):
    """``line: problem`` for every source that does not resolve and every
    decimal no source of its unit states."""
    problems = []
    for line, unit in units(text):
        named = sources(unit)
        for name, source in named:
            if source is None:
                problems.append(f'{line}: {name} does not resolve')
        texts = [source for _, source in named if source is not None]
        for number in decimals(unit):
            if not named:
                problems.append(f'{line}: {number} names no source')
            elif not any(_states(source, number) for source in texts):
                problems.append(f'{line}: {number} is in none of '
                                f'{[name for name, _ in named]}')
    return problems


def benchmark_tests():
    """``module::[Class::]test`` of every test in the benchmark modules."""
    found = []
    for path in sorted(_TESTS.glob('test_benchmarks_*.py')):
        for node in ast.parse(path.read_text(encoding='utf-8')).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test'):
                found.append(f'{path.name}::{node.name}')
            elif isinstance(node, ast.ClassDef):
                found += [f'{path.name}::{node.name}::{n.name}' for n in node.body
                          if isinstance(n, ast.FunctionDef)
                          and n.name.startswith('test')]
    return found


def test_every_number_on_the_validation_page_is_stated_by_its_source():
    problems = unbacked_numbers(PAGE.read_text(encoding='utf-8'))
    assert not problems, (
        'docs/models/validation.md quotes a number its cited test, table or '
        'section does not state:\n  ' + '\n  '.join(problems))


def test_every_benchmark_test_is_on_the_validation_page():
    named = {f'{m}{p}' for m, p in
             _TEST_ID.findall(PAGE.read_text(encoding='utf-8'))}
    missing = [t for t in benchmark_tests() if t not in named]
    assert not missing, ('benchmarks with no entry on '
                         'docs/models/validation.md:\n  ' + '\n  '.join(missing))


@pytest.mark.parametrize('unit, problem', [
    ('| RAM | 0.231 dB | `test_benchmarks_published.py::'
     'test_ram_matches_the_coupled_mode_reference_in_the_asa_wedge` |',
     '0.231 is in none of'),
    ('| RAM | 0.230 dB | `test_benchmarks_published.py::test_no_such_test` |',
     'does not resolve'),
    ('RAM sits 0.230 dB from the reference.', '0.230 names no source'),
    ('| RAM | 0.23 dB | `test_benchmarks_published.py::'
     'test_ram_matches_the_coupled_mode_reference_in_the_asa_wedge` |',
     '0.23 is in none of'),
])
def test_the_gate_reports_a_number_its_source_does_not_state(unit, problem):
    found = unbacked_numbers(unit)
    assert any(problem in line for line in found), found


@pytest.mark.parametrize('unit', [
    '| RAM | 0.230 dB, +4.35 dB | `test_benchmarks_published.py::'
    'test_ram_matches_the_coupled_mode_reference_in_the_asa_wedge`, '
    '`test_benchmarks_published.py::'
    'test_one_way_engines_lose_more_to_the_corrugation_than_the_two_way_reference` |',
    'Water over a 1704.5 m/s bottom (`benchmarks/corrugated_seafloor_25hz_couple07.txt`).',
    'A 25 Hz case at 200 m with no decimal names no source.',
])
def test_the_gate_accepts_a_number_its_source_states(unit):
    assert unbacked_numbers(unit) == []
