"""``uacpy.models.engine_status`` and ``python -m uacpy``: which binaries of
the registered engines an install holds, read through each engine's own
declared search (``PropagationModel._BINARIES``), and the exit status a
one-line install check returns."""

import os
import stat
import subprocess
import sys
import warnings

import pytest

import uacpy.models._launch as launch_mod
from uacpy.models import EngineStatus, engine_status
from uacpy.models._registry import ENGINES
from uacpy.tests.conftest import engine_params


def _status(**columns):
    """A one-engine-per-row status with every row found unless given."""
    n = len(columns['engine'])
    base = dict(backend=tuple(e.lower() for e in columns['engine']),
                binary=tuple(e.lower() for e in columns['engine']),
                install=('base',) * n, found=(True,) * n,
                path=('/bin/x',) * n, default=(True,) * n)
    base.update(columns)
    return EngineStatus(**base)


@pytest.fixture
def package_dir(tmp_path, monkeypatch):
    """An empty package directory as the search root, with nothing on
    ``PATH``: a binary is found only where a test puts one."""
    monkeypatch.setattr(launch_mod, '_PACKAGE_DIR', tmp_path)
    monkeypatch.setattr(launch_mod.shutil, 'which', lambda *a, **k: None)

    def install(subdir, name, executable=True):
        path = tmp_path / 'bin' / subdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#!/bin/sh\n')
        if executable:
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path
    return install


class TestTheTableListsEveryDeclaredBinary:

    def test_each_engine_has_one_row_per_declared_binary_in_registry_order(
            self, package_dir):
        status = engine_status()
        expected = [(entry.class_name, backend)
                    for entry in ENGINES.values()
                    for backend in entry.load()._BINARIES]
        assert list(zip(status.engine, status.backend)) == expected

    @pytest.mark.parametrize('cls', engine_params())
    def test_every_engine_declares_a_binary(self, cls):
        assert cls._BINARIES

    def test_the_oases_engines_are_the_optional_install(self, package_dir):
        status = engine_status()
        optional = {e for e, i in zip(status.engine, status.install)
                    if i == 'oases'}
        assert optional == {entry.class_name for entry in ENGINES.values()
                            if 'oases' in entry.requires}
        assert set(status.install) == {'base', 'oases'}

    def test_kraken_lists_its_field_binary_beside_its_backends(
            self, package_dir):
        status = engine_status()
        rows = [(b, n) for e, b, n in zip(status.engine, status.backend,
                                          status.binary) if e == 'Kraken']
        assert rows == [('kraken', 'kraken.exe'), ('krakenc', 'krakenc.exe'),
                        ('field', 'field.exe')]

    def test_to_dataframe_gives_one_row_per_binary(self, package_dir):
        pytest.importorskip('pandas')
        status = engine_status()
        frame = status.to_dataframe()
        assert list(frame.columns) == list(EngineStatus._TABLE_FIELDS)
        assert len(frame) == len(status.engine)

    def test_the_dict_round_trip_is_the_table(self, package_dir):
        status = engine_status()
        back = EngineStatus.from_dict(status.to_dict())
        assert repr(back) == repr(status)


class TestFoundMeansRunnableWhereTheEngineLooks:

    def test_an_executable_in_the_engine_s_bin_dir_is_found_there(
            self, package_dir):
        exe = package_dir('oalib', 'scooter.exe')
        status = engine_status()
        row = status.engine.index('Scooter')
        assert status.found[row] is True
        assert status.path[row] == str(exe)

    def test_a_file_without_its_execute_bit_is_not_found(self, package_dir):
        package_dir('oalib', 'scooter.exe', executable=False)
        status = engine_status()
        row = status.engine.index('Scooter')
        assert status.found[row] is False
        assert status.path[row] == ''

    def test_an_oases_binary_is_found_through_its_bash_wrapper(
            self, package_dir):
        package_dir('oases', 'oast')
        wrapper = package_dir('oases', 'oast_bash')
        status = engine_status()
        row = status.engine.index('OAST')
        assert status.path[row] == str(wrapper)
        assert status.binary[row] == 'oast'

    @pytest.mark.parametrize('cls', engine_params())
    def test_a_construction_resolves_the_binary_its_default_row_names(
            self, cls):
        """The declaration the table reads is the one the engine's own
        lookup reads: a construction runs the default row's binary."""
        entry = next(e for e in ENGINES.values()
                     if e.class_name == cls.__name__)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = cls(verbose=False, **dict(entry.example_kwargs))
        status = engine_status()
        rows = [i for i, (e, d) in enumerate(zip(status.engine,
                                                 status.default))
                if e == cls.__name__ and d]
        assert len(rows) == 1
        assert status.path[rows[0]] == str(model._exe)


class TestTheDefaultIsTheBackendAConstructionResolves:

    @pytest.mark.parametrize('installed, expected', [
        (('bellhopcuda', 'bellhopcxx', 'bellhop'), 'cuda'),
        (('bellhopcxx', 'bellhop'), 'cxx'),
        (('bellhop',), 'fortran'),
    ])
    def test_bellhop_marks_the_first_installed_of_cuda_cxx_fortran(
            self, package_dir, monkeypatch, installed, expected):
        from uacpy.models.bellhop import _backend
        monkeypatch.setattr(_backend, '_cuda_found_no_device', False)
        for name in installed:
            package_dir('oalib' if name == 'bellhop' else 'bellhopcuda', name)
        status = engine_status()
        marked = [b for e, b, d in zip(status.engine, status.backend,
                                       status.default)
                  if e == 'Bellhop' and d]
        assert marked == [expected]

    def test_no_bellhop_build_marks_no_default(self, package_dir):
        status = engine_status()
        assert not any(d for e, d in zip(status.engine, status.default)
                       if e == 'Bellhop')

    def test_kraken_and_ram_mark_the_backend_their_construction_resolves(
            self, package_dir):
        status = engine_status()
        marked = {e: b for e, b, d in zip(status.engine, status.backend,
                                          status.default) if d}
        assert marked['Kraken'] == 'kraken'
        assert marked['RAM'] == 'mpirams'


class TestMissingEngines:

    def test_a_base_engine_with_no_binary_found_is_missing(self):
        status = _status(engine=('Bellhop', 'Bellhop', 'Kraken'),
                         found=(False, False, True))
        assert status.missing_engines() == ('Bellhop',)

    def test_one_found_backend_is_enough(self):
        status = _status(engine=('Bellhop', 'Bellhop'), found=(False, True))
        assert status.missing_engines() == ()

    def test_an_optional_install_with_nothing_found_is_not_missing(self):
        status = _status(engine=('Kraken', 'OAST', 'OASN'),
                         install=('base', 'oases', 'oases'),
                         found=(True, False, False))
        assert status.missing_engines() == ()
        assert status.missing_engines(optional_installs=True) == (
            'OAST', 'OASN')

    def test_an_optional_install_with_one_binary_found_counts_each_engine(
            self):
        status = _status(engine=('Kraken', 'OAST', 'OASN'),
                         install=('base', 'oases', 'oases'),
                         found=(True, True, False))
        assert status.missing_engines() == ('OASN',)


class TestTheTablePrints:

    def test_each_row_shows_the_path_or_the_binary_not_found(self):
        status = _status(engine=('Bellhop', 'Bellhop'),
                         backend=('cuda', 'fortran'),
                         binary=('bellhopcuda', 'bellhop'),
                         found=(False, True), path=('', '/b/bellhop.exe'),
                         default=(False, True))
        assert repr(status) == (
            "EngineStatus(\n"
            "  engine   backend  default  found  path\n"
            "  Bellhop  cuda              no     not found: bellhopcuda\n"
            "  Bellhop  fortran  yes      yes    /b/bellhop.exe\n"
            ")")


def _run_main(*, package_dir=None):
    """``python -m uacpy`` in a fresh interpreter; with ``package_dir``, the
    search root is that directory and ``PATH`` is empty."""
    env = dict(os.environ)
    if package_dir is None:
        argv = [sys.executable, '-m', 'uacpy']
    else:
        env['PATH'] = ''
        argv = [sys.executable, '-c',
                "import pathlib, runpy, sys\n"
                "import uacpy.models._launch as launch\n"
                f"launch._PACKAGE_DIR = pathlib.Path({str(package_dir)!r})\n"
                "runpy.run_module('uacpy', run_name='__main__')\n"]
    return subprocess.run(argv, capture_output=True, text=True, env=env,
                          timeout=120)


class TestPythonMinusMUacpy:

    @pytest.mark.requires_binary
    def test_an_install_with_every_engine_found_exits_zero(self):
        proc = _run_main()
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert proc.stdout.startswith('uacpy ')
        assert 'EngineStatus(' in proc.stdout
        assert proc.stdout.rstrip().endswith(
            'Every installed engine has a binary.')

    def test_an_install_with_no_binary_exits_one_naming_the_engines(
            self, tmp_path):
        proc = _run_main(package_dir=tmp_path)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert 'No binary found for: Bellhop, Kraken' in proc.stdout
        assert 'Optional install not made: OAST' in proc.stdout

    def test_an_install_without_oases_exits_zero(self, tmp_path):
        """Every base binary installed, OASES not: the check passes and
        says which optional install was not made."""
        for entry in ENGINES.values():
            if 'oases' in entry.requires:
                continue
            for search in entry.load()._BINARIES.values():
                path = tmp_path / 'bin' / search.bin_subdirs[0] / \
                    search.names[0]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('#!/bin/sh\n')
                path.chmod(0o755)
        proc = _run_main(package_dir=tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert 'Optional install not made: OAST, OASN' in proc.stdout
