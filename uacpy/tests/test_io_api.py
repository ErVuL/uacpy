"""The public surface of ``uacpy.io``: the carriers the readers return,
one definition per unit helper, and one spelling of the file argument.
"""

import numpy as np
import pytest
import uacpy
from pathlib import Path


class TestTheCarriersTheReadersHandBackAreImportable:
    """A reader's return type is part of its contract, so it is exported too.

    ``read_reflection_coefficient`` returns the package's own
    :class:`~uacpy.ReflectionCoefficient`, so it reads, plots and exports as
    a Bounce or OASR table does; ``read_oast_tl`` returns a
    :class:`~uacpy.ResultStack` of TL fields over frequency.
    """

    TABLE = "3\n 0.0 1.0 0.0\n 45.0 0.5 3.14\n 90.0 0.1 0.0\n"

    def test_the_reflection_reader_returns_the_reflection_result(
            self, tmp_path):
        import uacpy.io as io

        path = tmp_path / 'table.brc'
        path.write_text(self.TABLE)
        table = io.read_reflection_coefficient(path)
        assert type(table) is uacpy.ReflectionCoefficient
        assert table.angles.tolist() == [0.0, 45.0, 90.0]
        assert table.magnitude.tolist() == [1.0, 0.5, 0.1]
        assert np.array_equal(table.phase, np.deg2rad([0.0, 3.14, 0.0]))
        assert table.frequencies is None
        assert table.phase_reference == 'travelling_wave'

    def test_no_reader_returns_a_dict_type_of_its_own(self):
        import uacpy.io as io

        for name in ('OastTL', 'ReflectionTable'):
            assert name not in io.__all__, name


class TestUnitHelpersHaveOneDefinition:
    """``km_to_m`` / ``m_to_km`` / ``deg_to_rad`` are pure arithmetic with no
    file format in them, so they live in :mod:`uacpy.core.units`;
    :mod:`uacpy.core.units` re-exports them under the same names and carries the
    file-format mandate.

    Two things have to hold for that split to be free: every existing
    ``from uacpy.core.units import ...`` still resolves, and the two module
    attributes are one object rather than two definitions free to drift."""

    NAMES = ('km_to_m', 'm_to_km', 'deg_to_rad')

    @pytest.mark.parametrize('name', NAMES)
    def test_the_io_name_is_the_core_object(self, name):
        import uacpy.core.units as core_units
        import uacpy.core.units as io_units
        assert getattr(io_units, name) is getattr(core_units, name)

    @pytest.mark.parametrize('name', NAMES)
    def test_exactly_one_definition_ships(self, name):
        import ast
        package = Path(uacpy.__file__).resolve().parent
        sites = []
        for path in sorted(package.rglob('*.py')):
            if 'third_party' in path.parts or 'tests' in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    sites.append(str(path.relative_to(package)))
        assert sites == [str(Path('core') / 'units.py')], sites

    def test_the_conversions_round_trip(self):
        from uacpy.core.units import deg_to_rad, km_to_m, m_to_km
        assert float(m_to_km(2000.0)) == 2.0
        assert float(km_to_m(2.0)) == 2000.0
        assert float(km_to_m(m_to_km(1234.5))) == pytest.approx(1234.5)
        assert float(deg_to_rad(180.0)) == pytest.approx(np.pi)

    def test_the_names_stay_off_the_io_public_surface(self):
        # They were never exported; the move must not add them.
        import uacpy.io as io_package
        assert not (set(self.NAMES) & set(io_package.__all__))


def test_every_io_reader_and_writer_names_its_file_filepath():
    """One spelling for the file argument of every public reader and
    writer. The block writers take an open file ``f``, and the two staging
    helpers take a table and a destination deck."""
    import inspect
    import uacpy.io
    exempt = {'stage_reflection_file', 'stage_source_beam_pattern'}
    offenders = []
    for name in uacpy.io.__all__:
        obj = getattr(uacpy.io, name)
        if (not inspect.isfunction(obj) or name in exempt
                or not name.startswith(('read_', 'write_', 'dedupe_'))):
            continue
        first = next(iter(inspect.signature(obj).parameters))
        if first not in ('filepath', 'f'):
            offenders.append(f"{name}({first})")
    assert not offenders


class TestTheIoSurfaceIsTheFormatsAUserNames:
    """``uacpy.io.__all__`` holds the readers and writers of files a user
    can name (decision 24). The deck blocks and staging helpers the model
    wrappers assemble decks from have one home each, the format module that
    defines them, and computation that is not format code lives where it
    belongs."""

    #: The deck blocks, by the format module that is their one home.
    _DECK_BLOCKS = {
        'uacpy.io.oalib_writer': (
            'write_header', 'write_absorption_block', 'write_fg_params',
            'write_bio_layers', 'write_broadband_freqs', 'write_ssp_section',
            'write_layer_sections', 'write_bottom_section', 'writable_layers',
            'write_source_depths', 'write_receiver_depths',
            'write_receiver_ranges', 'write_phase_speed_and_rmax',
            'resolve_ssp_interp', 'resolve_ssp_topopt'),
        'uacpy.io.refl_io': (
            'stage_reflection_file', 'stage_source_beam_pattern',
            'dedupe_reflection_file'),
    }

    def test_each_deck_block_is_defined_in_its_format_module(self):
        import importlib
        for home, names in self._DECK_BLOCKS.items():
            module = importlib.import_module(home)
            for name in names:
                assert getattr(module, name).__module__ == home, name

    def test_no_deck_block_is_on_the_public_surface(self):
        import uacpy.io
        names = {n for ns in self._DECK_BLOCKS.values() for n in ns}
        assert not names & set(uacpy.io.__all__)
        for name in names:
            assert not hasattr(uacpy.io, name), name

    def test_the_deck_blocks_have_no_second_import_path(self):
        import importlib
        with pytest.raises(ModuleNotFoundError, match='at_blocks'):
            importlib.import_module('uacpy.io.at_blocks')

    def test_the_moved_computation_is_reached_at_its_new_home(self):
        import uacpy.io
        from uacpy.core import _validate
        from uacpy.models import sparc
        for name in ('equally_spaced', 'rts_to_pressure'):
            assert name not in uacpy.io.__all__
            assert not hasattr(uacpy.io, name), name
        assert _validate.equally_spaced([0.0, 1.0, 2.0]) is True
        assert 'rts_to_pressure' in sparc.__all__
        assert (sparc.rts_to_pressure.__module__
                == 'uacpy.models.sparc._extract')

    def test_the_sparc_source_series_writer_is_public(self):
        """A companion file of a deck, as ``write_inpe``'s four are: a user
        running SPARC on a deck of their own needs it beside the deck."""
        import inspect
        import uacpy.io
        from uacpy.io import oalib_writer
        assert 'write_sparc_source_time_series' in uacpy.io.__all__
        assert (uacpy.io.write_sparc_source_time_series
                is oalib_writer.write_sparc_source_time_series)
        first = next(iter(inspect.signature(
            uacpy.io.write_sparc_source_time_series).parameters))
        assert first == 'filepath'
