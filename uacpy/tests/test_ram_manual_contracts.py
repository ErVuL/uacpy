"""Pins for the RAM/PE family's own documentation — the vendor contracts
uacpy's wrappers must honor, where no other test file already does.

Sources pinned here:

- ``third_party/mpiramS/README`` / ``README.OMP`` / ``README.RECL`` and the
  rewritten ``src/peramx.f90`` that supersedes the stock ``in.pe`` layout
  (``third_party/MODIFICATIONS.md``): record order of ``in.pe``, the OpenMP
  thread contract, the stack-limit note.
- ``third_party/ramsurf/readme.orig`` (Calvo's RAM/RAMS guide) and
  ``README.rst`` (the C port, which spells out the ramsurf surface-block
  position): ``ram.in`` row/block order, ``rs = 0`` semantics, attenuation
  in dB/wavelength.
- The RAM manual (RAM.md — Collins, *User's Guide for RAM*): stability
  constraints ``ns``/``rs`` and their defaults, sediment attenuation units.
- ``third_party/ramgeo/README.md`` + the three vendored Collins sources:
  the input-deck filename each binary hardcodes.
- ``third_party/bellhopcuda/README.md`` + ``doc/accuracy.md`` and the
  match/nomatch fixture lists: where cxx/cuda output is expected to diverge
  from Fortran.

Writer- and parser-level tests are binary-free; classes that construct
``RAM`` (which resolves its binary) carry ``requires_binary``.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest

from uacpy.core.environment import (
    BoundaryProperties,
    Environment,
    SeabedColumn,
    SedimentLayer,
)
from uacpy.core.receiver import Receiver
from uacpy.core.source import Source
from uacpy.io.mpirams_writer import write_inpe
from uacpy.io.ramsurf_writer import write_ramin
from uacpy.models.ram import mpirams as ram_mpirams, collins as ram_collins
from uacpy.tests.conftest import make_pekeris

THIRD_PARTY = Path(__file__).resolve().parents[1] / 'third_party'


def _fluid_env():
    return make_pekeris(name='manual', sound_speed=1600.0, density=1.5)


# ─── in.pe record order: writer vs the peramx.f90 this build reads ─────────


class TestInPeRecordOrderMatchesPeramx:
    """``mpiramS/README`` (d) warns the ``in.pe`` read is positional. For
    uacpy's build the authority is the rewritten ``peramx.f90``
    (``third_party/MODIFICATIONS.md`` — the stock layout has a receiver-range
    line and node-weight records this build does not read, and lacks the
    ``c0_user`` / ranges-file / sediment records it does). One side parses
    the read sequence out of the vendored source; the other checks the deck
    ``write_inpe`` emits slot by slot. Existing tests pin only slot 6
    (``np nss``); a swap anywhere else would feed every later record to the
    wrong variable and still march."""

    # Scalar-record read order of peramx.f90's in.pe block, in the variable
    # names the source uses. range_dependent_sediment=0 appends three nzs-value array records;
    # range_dependent_sediment=1 appends the sediment filename instead.
    EXPECTED_READS = [
        'dum', 'fc, Q', 'T', 'zsrc(1)', 'deltaz', 'deltar', 'np, nss',
        'rs', 'dzm', 'c0_user', 'name1', 'iflat', 'ihorz', 'ibot',
        'name2', 'name3', 'sedlayer', 'nzs', 'isedrd',
    ]

    def test_the_vendored_source_reads_the_documented_order(self):
        src_path = THIRD_PARTY / 'mpiramS' / 'src' / 'peramx.f90'
        if not src_path.exists():
            pytest.skip("peramx.f90 not vendored here")
        src = src_path.read_text(errors='ignore')
        block = src[src.index("open(nunit,file='in.pe'"):]
        block = block[:block.index('if (isedrd==1)')]
        reads = [v.strip() for _, v in re.findall(
            r"read \(nunit,(\*|'\(a\)')\)\s*([^!\n]+)", block)]
        assert reads == self.EXPECTED_READS

    @staticmethod
    def _deck(tmp_path, **overrides):
        kwargs = dict(
            fc=111.0, q_factor=22.0, record_duration=33.0, zsrc=44.0, dz=0.55, dr=66.0,
            n_pade=7, n_stability=2, stability_range_m=88.0, depth_decimation=9, ssp_filename='S.ssp',
            earth_curvature=1, horizontal_interpolation=0, bathymetry_from_file=1, bth_filename='B.bth',
            sedlayer=123.0, n_sediment_points=5,
            cs=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            rho=np.full(5, 1.5), attn=np.full(5, 0.25),
            c0=1234.0,
        )
        kwargs.update(overrides)
        path = tmp_path / 'in.pe'
        write_inpe(path, **kwargs)
        return path.read_text().splitlines()

    def test_every_scalar_record_lands_in_its_read_slot(self, tmp_path):
        # Sentinel values all distinct, so any record swap moves a checked
        # number. Slot indices are 0-based lines of the deck; slots 1 and 6
        # carry two values each, exactly as the paired reads expect.
        lines = self._deck(tmp_path)
        assert len(lines) == len(self.EXPECTED_READS) + 3
        assert lines[1].split() == ['111.0', '22.0']       # fc, Q
        assert float(lines[2]) == 33.0                     # T
        assert float(lines[3]) == 44.0                     # zsrc
        assert float(lines[4]) == 0.55                     # deltaz
        assert float(lines[5]) == 66.0                     # deltar
        assert lines[6].split() == ['7', '2']              # np, nss
        assert float(lines[7]) == 88.0                     # rs
        assert int(lines[8]) == 9                          # dzm
        assert float(lines[9]) == 1234.0                   # c0_user
        assert lines[10] == 'S.ssp'                        # name1
        assert [lines[11], lines[12], lines[13]] == ['1', '0', '1']
        assert lines[14] == 'B.bth'                        # name2
        assert lines[15] == 'ranges.dat'                   # name3
        assert float(lines[16]) == 123.0                   # sedlayer
        assert int(lines[17]) == 5                         # nzs
        assert int(lines[18]) == 0                         # isedrd
        assert [float(v) for v in lines[19].split()] == [1, 2, 3, 4, 5]
        assert all(float(v) == 1.5 for v in lines[20].split())
        assert all(float(v) == 0.25 for v in lines[21].split())

    def test_range_dependent_sediment_swaps_arrays_for_the_filename(
            self, tmp_path):
        lines = self._deck(tmp_path, range_dependent_sediment=1, sed_filename='sed.dat')
        assert int(lines[18]) == 1
        assert lines[19] == 'sed.dat'
        assert len(lines) == 20


# ─── Collins deck filenames vs the vendored OPEN statements ────────────────


class TestCollinsDeckFilenamesMatchTheVendoredOpens:
    """Each Collins binary hardcodes its own input filename (``ramgeo``
    README: "Reads ``ramgeo.in``"; ``readme.orig``: "ram.in (the input to
    RAM)", "rams.in (the input to RAMS)"). ``ram.collins.collins_in_name``
    carries the mapping; parse the ``open(unit=1,...)`` out of each vendored
    source so a vendor refresh that renames a deck fails here, not as a hung
    binary waiting on a file that was never written."""

    SOURCES = {
        'ramgeo': ('ramgeo', 'ramgeo1.5.f'),
        'ramsurf': ('ramsurf', 'ramsurf1.5.f'),
        'rams': ('ramsurf', 'rams0.5.f'),
    }

    @pytest.mark.parametrize('kind', sorted(SOURCES))
    def test_in_name_matches_the_sources_open_statement(self, kind):
        subdir, fname = self.SOURCES[kind]
        path = THIRD_PARTY / subdir / fname
        if not path.exists():
            pytest.skip(f"{fname} not vendored here")
        m = re.search(r"open\(unit=1,status='old',file='([^']+)'\)",
                      path.read_text(errors='ignore'))
        assert m, f"no input-deck OPEN in {fname}"
        assert ram_collins.collins_in_name(kind) == m.group(1)


# ─── Stability constraints: default decks keep them on all the way ─────────


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
class TestStabilityConstraintDefaultsSpanTheMarch:
    """RAM manual §3: "When rs is set to 0, the stability constraints are
    used for all ranges", and ``ns = 1 or 2`` is effective for most problems
    (§2); ``readme.orig`` row 5: "For radius of stability constraint, leave
    at 0". mpiramS spells the same intent differently — ``ram.f90:68``
    arms the cutoff as ``rsc = |rend-rnow| - rs``, so constraints stay on
    until ``rs`` of range is covered and ``rs = rmax`` keeps them on for
    the whole march. uacpy's defaults must encode "on everywhere" on both
    deck families."""

    def _capture(self, monkeypatch, writer_name, backend, run):
        captured = {}

        def fake(*args, **kwargs):
            captured.update(kwargs)
            raise RuntimeError('deck captured')

        writer_module = {'write_inpe': ram_mpirams,
                         'write_ramin': ram_collins}[writer_name]
        monkeypatch.setattr(writer_module, writer_name, fake)
        from uacpy.models.ram import RAM
        model = RAM(verbose=False, dr=20.0, dz=2.0, backend=backend)
        with pytest.raises(RuntimeError, match='deck captured'):
            run(model)
        return captured

    def test_mpirams_default_rs_is_the_march_end(self, monkeypatch):
        env = _fluid_env()
        src = Source(depths=50.0, frequencies=50.0)
        rcv = Receiver(depths=np.array([50.0]), ranges=np.array([3000.0]))
        captured = self._capture(
            monkeypatch, 'write_inpe', None,
            lambda m: m.compute_tl(env=env, source=src, receiver=rcv))
        assert captured['stability_range_m'] == pytest.approx(3000.0)
        assert captured['n_stability'] in (1, 2)

    def test_collins_default_row5_is_ns1_rs0(self, monkeypatch):
        env = _fluid_env()
        src = Source(depths=50.0, frequencies=50.0)
        rcv = Receiver(depths=np.array([50.0]), ranges=np.array([3000.0]))
        captured = self._capture(
            monkeypatch, 'write_ramin', 'ramgeo',
            lambda m: m.compute_tl(env=env, source=src, receiver=rcv))
        assert captured['n_stability'] == 1
        assert captured['stability_range_m'] == 0.0


# ─── README.OMP: thread count and stack limit ──────────────────────────────


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
class TestOmpThreadContract:
    """``README.OMP``: "The openmp standard is that the variable
    OMP_NUM_THREADS sets the number of parallel threads." The wrapper passes
    an explicit user setting through to the mpiramS child as given; inside a
    ``run_parallel`` worker it pins one thread, so the pool does not
    multiply into cpu_count² threads; otherwise it leaves the variable
    unset and the child uses the cores (the rule Bellhop's ``-1`` follows,
    ``_launch.openmp_thread_env``)."""

    def _spawn_env(self, monkeypatch, tmp_path):
        from uacpy.models.ram import RAM
        model = RAM(verbose=False)
        seen = {}

        def fake_run(self, cmd, cwd, timeout=None, stdin_input=None,
                     env=None, check=True):
            seen['env'] = env
            raise RuntimeError('spawn captured')

        monkeypatch.setattr(RAM, '_run_subprocess', fake_run)
        with pytest.raises(RuntimeError, match='spawn captured'):
            model._run_binary(tmp_path)
        return seen['env']

    @staticmethod
    def _in_pool(monkeypatch, in_pool):
        from uacpy.models import _launch
        monkeypatch.setattr(_launch, '_single_threaded_launches', in_pool)

    def test_unset_outside_a_pool_leaves_the_cores_to_the_child(
            self, monkeypatch, tmp_path):
        monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
        self._in_pool(monkeypatch, False)
        assert 'OMP_NUM_THREADS' not in self._spawn_env(monkeypatch, tmp_path)

    def test_unset_in_a_pool_worker_is_one_thread(self, monkeypatch,
                                                  tmp_path):
        monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
        self._in_pool(monkeypatch, True)
        assert self._spawn_env(monkeypatch, tmp_path)['OMP_NUM_THREADS'] == '1'

    @pytest.mark.parametrize('in_pool', [False, True])
    def test_a_user_setting_is_inherited_verbatim(self, monkeypatch,
                                                  tmp_path, in_pool):
        monkeypatch.setenv('OMP_NUM_THREADS', '7')
        self._in_pool(monkeypatch, in_pool)
        assert self._spawn_env(monkeypatch, tmp_path)['OMP_NUM_THREADS'] == '7'


class TestStackLimitIsRaisedForTheFortranChildren:
    """``README.OMP``: "gfortran and ifort seem to need the ulimit to be set
    larger, otherwise eigenray just segfaults" (its ``ulimit -s unlimited``
    line). Every spawned binary runs with its soft RLIMIT_STACK at the hard
    limit, raised inside the child by ``uacpy._stack.stack_limit_prefix``;
    the importing process keeps its own limit, and ``UACPY_NO_STACK_RAISE``
    opts out."""

    @staticmethod
    def _lowered_soft(hard):
        import resource
        low = 8 << 20
        if hard != resource.RLIM_INFINITY:
            low = min(low, hard)
        return low

    @staticmethod
    def _expected_kib(hard):
        import resource
        return ('unlimited' if hard == resource.RLIM_INFINITY
                else str(hard // 1024))

    def _child_ulimit(self, tmp_path):
        from uacpy.models import RAM
        out = RAM(verbose=False, timeout=30.0)._run_subprocess(
            ['/bin/sh', '-c', 'ulimit -s'], cwd=tmp_path)
        return out.stdout.strip()

    def test_importing_the_models_leaves_the_process_limit_alone(self):
        resource = pytest.importorskip('resource')
        import subprocess
        import sys
        code = ('import resource; a = resource.getrlimit(resource.RLIMIT_STACK);'
                ' import uacpy.models, uacpy;  uacpy.Bellhop;'
                ' b = resource.getrlimit(resource.RLIMIT_STACK);'
                ' print(a == b)')
        _soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        low = self._lowered_soft(hard)
        out = subprocess.run(
            ['/bin/sh', '-c', f'ulimit -S -s {low // 1024}; exec "$0" "$@"',
             sys.executable, '-c', code],
            capture_output=True, text=True, check=True,
            env={k: v for k, v in os.environ.items()
                 if k != 'UACPY_NO_STACK_RAISE'})
        assert out.stdout.strip().splitlines()[-1] == 'True'

    @pytest.mark.requires_binary
    def test_a_spawned_binary_runs_at_the_hard_limit(self, monkeypatch,
                                                      tmp_path):
        resource = pytest.importorskip('resource')
        soft0, hard = resource.getrlimit(resource.RLIMIT_STACK)
        monkeypatch.delenv('UACPY_NO_STACK_RAISE', raising=False)
        low = self._lowered_soft(hard)
        if low == hard:
            pytest.skip('hard stack limit is already the 8 MiB default')
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (low, hard))
            assert self._child_ulimit(tmp_path) == self._expected_kib(hard)
            assert resource.getrlimit(resource.RLIMIT_STACK)[0] == low
        finally:
            resource.setrlimit(resource.RLIMIT_STACK, (soft0, hard))

    @pytest.mark.requires_binary
    def test_opt_out_leaves_the_child_at_the_inherited_limit(
            self, monkeypatch, tmp_path):
        resource = pytest.importorskip('resource')
        soft0, hard = resource.getrlimit(resource.RLIMIT_STACK)
        monkeypatch.setenv('UACPY_NO_STACK_RAISE', '1')
        low = self._lowered_soft(hard)
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (low, hard))
            assert self._child_ulimit(tmp_path) == str(low // 1024)
        finally:
            resource.setrlimit(resource.RLIMIT_STACK, (soft0, hard))


# ─── ram.in block order (readme.orig rows / README.rst layout) ─────────────


class TestRamInBlockOrderIsTheDocumentedOne:
    """``readme.orig`` fixes the section order — rows 1-5, bathymetry, then
    per range: water SSP, bottom sound speed, density, attenuation, with a
    bare range line starting each later section; ``README.rst`` places the
    ramsurf surface block between row 5 and the bathymetry; for RAMS the
    shear-speed block follows the compressional speed and shear attenuation
    follows compressional attenuation (``rams0.5.f:191-196``'s six ``zread``
    calls). Existing writer tests count ``-1 -1`` terminators, which cannot
    see a block swap — these read the deck back with per-block sentinel
    values."""

    @staticmethod
    def _blocks(lines):
        """(depth, value) blocks of the deck body, split on ``-1 -1``."""
        blocks, current = [], []
        for line in lines:
            fields = line.split()
            if len(fields) != 2:
                continue
            if fields[0] == '-1':
                blocks.append(current)
                current = []
                continue
            current.append((float(fields[0]), float(fields[1])))
        return blocks

    def test_ramsurf_surface_sits_between_row5_and_bathymetry(self, tmp_path):
        out = tmp_path / 'ram.in'
        write_ramin(
            str(out), kind='ramsurf', frequency=100.0, zs=50.0, zr_line=60.0,
            rmax_march=5000.0, dr=10.0, ndr=2, zmax=400.0, dz=1.0, depth_decimation=2,
            zmplt=200.0, c0=1500.0, n_pade=4,
            surface=[(0.0, 1.25)],
            bathymetry=[(0.0, 101.0)],
            range_segments=[
                dict(range=0.0, water_ssp=[(0.0, 1501.0)],
                     bottom_c=[(0.0, 1601.0)], bottom_rho=[(0.0, 1.61)],
                     bottom_attn=[(0.0, 0.61)]),
                dict(range=2500.0, water_ssp=[(0.0, 1502.0)],
                     bottom_c=[(0.0, 1602.0)], bottom_rho=[(0.0, 1.62)],
                     bottom_attn=[(0.0, 0.62)]),
            ])
        lines = out.read_text().splitlines()
        blocks = self._blocks(lines[5:])
        assert [b[0][1] for b in blocks] == [
            1.25,                          # surface — first block after row 5
            101.0,                         # bathymetry
            1501.0, 1601.0, 1.61, 0.61,    # segment 1: cw, cb, rho, attn
            1502.0, 1602.0, 1.62, 0.62,    # segment 2, same order
        ]
        # The bare range line divides the two segments, after segment 1's
        # attn terminator and before segment 2's SSP.
        assert lines.index('2500') == lines.index('0 0.61') + 2

    def test_rams_shear_blocks_interleave_after_cb_and_attn(self, tmp_path):
        out = tmp_path / 'rams.in'
        write_ramin(
            str(out), kind='rams', frequency=100.0, zs=50.0, zr_line=60.0,
            rmax_march=5000.0, dr=10.0, ndr=2, zmax=400.0, dz=1.0, depth_decimation=2,
            zmplt=200.0, c0=1500.0, n_pade=4, rams_rotation=1, rams_rotation_angle=60.0,
            bathymetry=[(0.0, 101.0)],
            range_segments=[dict(
                range=0.0, water_ssp=[(0.0, 1501.0)],
                bottom_c=[(0.0, 1601.0)], bottom_cs=[(0.0, 401.0)],
                bottom_rho=[(0.0, 1.61)], bottom_attn=[(0.0, 0.61)],
                bottom_attns=[(0.0, 0.91)])])
        lines = out.read_text().splitlines()
        assert lines[4].split() == ['1500', '4', '1', '60']  # c0 np irot theta
        blocks = self._blocks(lines[5:])
        assert [b[0][1] for b in blocks] == [
            101.0,                                    # bathymetry (no surface)
            1501.0, 1601.0, 401.0, 1.61, 0.61, 0.91,  # cw cb cs rho attn attns
        ]


# ─── Sediment attenuation is dB/wavelength (frequency-independent) ─────────


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
class TestCollinsAttnIsPerWavelength:
    """RAM manual Fig. 3: ``attn`` — "attenuation in sediment
    (dB/wavelength)"; ``readme.orig`` warns dB/(m·kHz) tables must be
    converted before entry. uacpy's seabed attenuation is dB/λ throughout,
    so the deck value is the environment's number verbatim and does not
    scale with the run frequency — which is what a mistaken dB/(m·kHz)
    interpretation would do."""

    def _attn_head(self, freq):
        from uacpy.models.ram import RAM
        env = Environment(
            name='attn', bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=15, sound_speed=1650,
                                      density=1.6, attenuation=0.4)],
                halfspace=BoundaryProperties(acoustic_type='half-space',
                                             sound_speed=1900, density=1.9,
                                             attenuation=0.2)))
        model = RAM(verbose=False)
        seg = ram_collins.collins_range_segments(
            env, 'ramgeo', zmax=400.0, freq=freq, knobs=model._knob_record(),
            speed_bounds=model._speed_bounds)[0]
        return seg['bottom_attn'][:2]

    def test_deck_attenuation_does_not_scale_with_frequency(self):
        lo, hi = self._attn_head(25.0), self._attn_head(400.0)
        assert lo == hi
        # And the value is the environment's own dB/λ number.
        assert lo[0][1] == pytest.approx(0.4)


# ─── bellhopcuda: the vendored divergence expectations ─────────────────────


class TestBellhopcudaDivergenceFixtures:
    """``bellhopcuda/doc/accuracy.md`` compares the ports to the modified
    Fortran and records which cases are expected to diverge; the
    ``*_match.txt`` / ``*_nomatch.txt`` lists are that record. uacpy's
    Bellhop treats cxx/cuda/Fortran as interchangeable 2D backends
    (``models/bellhop/``) and refuses 3D outright, which is sound exactly
    while the vendored suite shows 2D TL fully matching and confines the
    expected divergences to 3D/Nx2D plus one broadband-arrivals case. A
    vendor refresh that grows those lists must fail here so the wrapper's
    accuracy story gets re-examined."""

    DIR = THIRD_PARTY / 'bellhopcuda'

    @classmethod
    def _cases(cls, name):
        path = cls.DIR / name
        lines = path.read_text().splitlines()
        return [ln.strip() for ln in lines
                if ln.strip() and not ln.strip().startswith('//')]

    def _need_fixtures(self):
        if not (self.DIR / 'tl_match.txt').exists():
            pytest.skip("bellhopcuda fixture lists not vendored here")

    def test_2d_tl_has_a_match_list_and_no_nomatch_list(self):
        self._need_fixtures()
        assert len(self._cases('tl_match.txt')) >= 40
        assert not (self.DIR / 'tl_nomatch.txt').exists()

    def test_expected_divergences_are_3d_family_or_one_arrivals_case(self):
        self._need_fixtures()
        # The only nomatch lists shipped are for run modes the wrapper
        # cannot emit (3D / Nx2D, rejected via dimensionality='2D' only)...
        nomatch_files = sorted(
            p.name for p in self.DIR.glob('*_nomatch.txt'))
        assert nomatch_files == [
            'arrivals_nomatch.txt', 'tl3d_nomatch.txt',
            'tl_Nx2D_nomatch.txt',
        ]
        # ...plus exactly one reachable 2D case: the broadband surface-duct
        # arrivals run. Anything more means new documented divergence in
        # territory the wrapper can reach.
        assert self._cases('arrivals_nomatch.txt') == ['sduct_bbB']
