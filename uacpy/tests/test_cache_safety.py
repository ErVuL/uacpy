"""The offline cache must not execute what it reads, nor write where it is told.

Two findings, both demonstrated before they were fixed:

* the EMODnet polygon index and the NSIDC sea-ice climatology were **pickles**,
  so a planted cache file ran arbitrary code through ``fetch_bottom_local`` and
  ``fetch_sea_ice_concentration``. Neither file is ever downloaded — both are
  built locally — so the vector is write access to the cache directory, which a
  shared ``$UACPY_DATA_CACHE`` and a tarball'd working tree both hand over.
* every install-time writer staged at a predictable ``<out>.part``, so a
  symlink pre-placed there sent the run's output wherever the link pointed and
  left the cache entry a symlink, redirecting every later write too.

The tests below assert the *safe* behaviour — a payload does not run, a
pre-placed link does not redirect — not merely that the happy path still works.
"""

import os
import re
from pathlib import Path

import numpy as np
import pytest

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import _cache, emodnet_local, seaice_local

_DATA_DIR = Path(_cache.__file__).parent


class _Payload:
    """Unpickling/loading this writes a marker file, standing in for any code."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (os.system, (f"echo ran > {self.marker}",))


def _seaice_npz(path, marker=None):
    """A sea-ice climatology npz — real arrays, or an object array if payloaded."""
    if marker is None:
        arrays = {'N': np.full((12, 4, 4), 0.6, np.float32),
                  'S': np.full((12, 4, 4), 0.6, np.float32)}
    else:
        arrays = {h: np.array([_Payload(marker)], dtype=object) for h in 'NS'}
    np.savez_compressed(path, **arrays)


def _emodnet_npz(path, wkb, codes):
    np.savez_compressed(
        path, codes=np.asarray(codes, dtype=np.int32),
        wkb=np.frombuffer(b''.join(wkb), dtype=np.uint8),
        offsets=np.cumsum([0] + [len(b) for b in wkb], dtype=np.int64))


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty ``$UACPY_DATA_CACHE`` with both memos dropped either side."""
    root = tmp_path / 'data_cache'
    (root / 'emodnet').mkdir(parents=True)
    (root / 'seaice').mkdir(parents=True)
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    _cache.invalidate_grids()
    yield root
    _cache.invalidate_grids()


# ── the cache must not execute what it reads ──────────────────────────────────

def test_a_planted_object_array_is_refused_without_running_its_payload(
        cache, tmp_path):
    """The npz reader passes ``allow_pickle=False``, so an object array in a
    cache file is refused at load rather than reconstructed — which is what
    would run the code it encodes."""
    marker = tmp_path / 'ran.txt'
    _seaice_npz(cache / 'seaice' / seaice_local.INDEX_FILE, marker=marker)
    with pytest.raises(DataFetchError, match='present but unreadable'):
        seaice_local.fetch_sea_ice_concentration((80.0, 0.0), month=3)
    assert not marker.exists()


def test_every_cache_np_load_pins_allow_pickle_false():
    """numpy defaults ``allow_pickle`` to False today, so no behavioural test
    can tell a dropped argument from a kept one — but the default is the thing
    that changed in numpy 1.16.3 after it was found exploitable, and these
    files are read straight out of a directory an attacker may write. The
    argument is therefore pinned in the source."""
    bare = []
    for path in sorted(_DATA_DIR.glob('*.py')):
        for call in re.findall(r'np\.load\([^)]*\)', path.read_text()):
            if 'allow_pickle=False' not in call:
                bare.append(f"{path.name}: {call}")
    assert not bare, f"np.load without an explicit allow_pickle=False: {bare}"


def test_the_readers_import_no_pickle():
    """The two converted readers must not regain a pickle path by any route."""
    for module in (emodnet_local, seaice_local):
        body = Path(module.__file__).read_text()
        assert not re.search(r'^\s*import pickle', body, re.M), module.__name__
        assert 'pickle.load' not in body, module.__name__


# ── a leftover pickle is refused, not converted ───────────────────────────────

@pytest.mark.parametrize('module, dataset, flag', [
    (emodnet_local, 'emodnet', './install.sh --data emodnet'),
    (seaice_local, 'seaice', './install.sh --data seaice'),
])
def test_a_leftover_pickle_index_is_refused_with_a_migration_error(
        module, dataset, flag, cache, tmp_path):
    """Converting the old file in place would have to unpickle it first, which
    is the execution the format change removes — so it is refused. The message
    must say *why* (a security migration, not a damaged file) and name both the
    file to delete and the flag that rebuilds it."""
    marker = tmp_path / 'ran.txt'
    import pickle
    stale = cache / dataset / module.RETIRED_INDEX_FILE
    stale.write_bytes(pickle.dumps(_Payload(marker)))

    with pytest.raises(ConfigurationError) as excinfo:
        module._build_index() if dataset == 'emodnet' else module._build_model()

    text = str(excinfo.value)
    assert not marker.exists(), 'the refusal unpickled the file it refused'
    assert str(stale) in text                       # names the offending file
    assert flag in text                             # names the install flag
    assert 'security migration' in text
    assert 'not a damaged file' in text
    assert 'corrupt' not in text.lower()


def test_a_missing_dataset_reports_not_found_without_the_migration_wording(cache):
    """With no index of either format the message is the ordinary not-found
    one; the migration wording must not leak onto a cold cache."""
    with pytest.raises(ConfigurationError, match='not found') as excinfo:
        emodnet_local._build_index()
    assert 'migration' not in str(excinfo.value)


def test_a_present_npz_wins_over_a_leftover_pickle(cache):
    """A rebuilt cache that still has the old file beside the new one is
    already migrated: the refusal is for caches with *only* the pickle."""
    (cache / 'emodnet' / emodnet_local.RETIRED_INDEX_FILE).write_bytes(b'stale')
    shapely = pytest.importorskip('shapely')
    poly = shapely.geometry.box(2.0, 54.0, 3.0, 56.0)
    _emodnet_npz(cache / 'emodnet' / emodnet_local.INDEX_FILE,
                 [shapely.to_wkb(poly)], [2])
    assert emodnet_local.fetch_seabed_local((55.0, 2.5))['folk_5cl'] == 2


# ── the round-trip the format change rests on ────────────────────────────────

def test_the_npz_index_round_trips_geometry_codes_and_lookup(cache):
    """Three arrays (codes, one concatenated WKB blob, offsets) must rebuild
    the same geometries, the same codes and the same point-in-polygon answer
    the pickled list of WKB bytes gave."""
    shapely = pytest.importorskip('shapely')
    polys = [shapely.geometry.box(2.0, 54.0, 3.0, 56.0),      # 86 B-ish
             shapely.geometry.Point(6.0, 58.0).buffer(0.5)]   # many vertices
    wkb = [shapely.to_wkb(p) for p in polys]
    _emodnet_npz(cache / 'emodnet' / emodnet_local.INDEX_FILE, wkb, [2, 5])

    tree, codes = emodnet_local._index()
    assert codes.tolist() == [2, 5]
    rebuilt = [tree.geometries[i] for i in range(len(polys))]
    assert [shapely.to_wkb(g) for g in rebuilt] == wkb        # byte-identical
    assert all(shapely.equals_exact(a, b, 0.0)
               for a, b in zip(polys, rebuilt))
    assert emodnet_local.fetch_seabed_local((55.0, 2.5))['folk_5cl'] == 2
    assert emodnet_local.fetch_seabed_local((58.0, 6.0))['folk_5cl'] == 5


def test_the_seaice_npz_round_trips_nan_and_dtype(cache):
    """Land is NaN and the grids are float32; a container change must not
    quietly promote the dtype or fill the holes."""
    grid = np.full((12, 4, 4), 0.6, np.float32)
    grid[:, 0, 0] = np.nan
    np.savez_compressed(cache / 'seaice' / seaice_local.INDEX_FILE,
                        N=grid, S=grid)
    back = seaice_local.sea_ice_grid(3, hemi='N')
    assert back.dtype == np.float32
    assert np.array_equal(back, grid[2], equal_nan=True)
    assert np.isnan(back[0, 0])


# ── staging must not follow a pre-placed symlink ─────────────────────────────

def test_a_symlink_at_the_old_staging_name_does_not_redirect_the_write(
        tmp_path):
    """The pre-placed ``<out>.part`` link must be left untouched: the content
    lands in the cache, the victim file outside it keeps its bytes, and the
    published cache entry is a regular file rather than the link itself."""
    cache_dir = tmp_path / 'cache'; cache_dir.mkdir()
    outside = tmp_path / 'outside'; outside.mkdir()
    victim = outside / 'victim.bin'
    victim.write_bytes(b'ORIGINAL')
    out = cache_dir / 'index.npz'
    os.symlink(victim, str(out) + '.part')

    with _cache.atomic_write(out) as part:
        part.write_bytes(b'RUN OUTPUT')

    assert victim.read_bytes() == b'ORIGINAL'      # not written through
    assert not out.is_symlink()                    # not published as a link
    assert out.read_bytes() == b'RUN OUTPUT'


def test_the_staging_name_is_not_predictable_from_the_destination(tmp_path):
    """Two writes of the same destination must stage under different names —
    the property that leaves an attacker nothing to pre-place a link at."""
    out = tmp_path / 'index.npz'
    seen = set()
    for _ in range(4):
        with _cache.atomic_write(out) as part:
            seen.add(Path(part).name)
            assert Path(part).name != out.name + '.part'
            part.write_bytes(b'x')
    assert len(seen) == 4


def test_a_staging_file_is_not_mistaken_for_a_cached_grid(tmp_path):
    """``gebco_local._grid`` picks its grid by globbing ``*.nc``, so a staging
    file that kept the destination's extension would be sampled as one."""
    out = tmp_path / 'GEBCO_2025.nc'
    with _cache.atomic_write(out) as part:
        assert list(tmp_path.glob('*.nc')) == []
        part.write_bytes(b'grid')
    assert list(tmp_path.glob('*.nc')) == [out]


def test_staging_keeps_the_mode_a_plain_create_would_have_given(tmp_path):
    """mkstemp creates 0600, but a cache directory may be shared and these are
    public reference grids: tightening them silently would change who can read
    an existing cache."""
    out = tmp_path / 'index.npz'
    with _cache.atomic_write(out) as part:
        part.write_bytes(b'x')
    reference = tmp_path / 'reference.bin'
    reference.write_bytes(b'x')
    assert (out.stat().st_mode & 0o777) == (reference.stat().st_mode & 0o777)


def test_no_python_writer_stages_at_a_predictable_part_name():
    """Five readers rolled their own ``<out>.part`` inline rather than calling
    ``atomic_write``; each was reachable by the same pre-placed link."""
    offenders = []
    for path in sorted(_DATA_DIR.glob('*.py')):
        for line in path.read_text().splitlines():
            if re.search(r"""\+ ['"]\.part['"]""", line):
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, f"predictable .part staging: {offenders}"


def test_curl_download_stages_off_the_predictable_name(tmp_path, monkeypatch):
    """``curl -o`` follows a symlink at the name it is given, and cannot route
    through ``atomic_write`` because its failure path is a ``False`` return."""
    from uacpy.data import _http
    victim = tmp_path / 'victim.bin'
    victim.write_bytes(b'ORIGINAL')
    out = tmp_path / 'grid.nc'
    os.symlink(victim, str(out) + '.part')

    def fake_run(argv, **kwargs):
        Path(argv[argv.index('-o') + 1]).write_bytes(b'DOWNLOADED')
        return None

    monkeypatch.setattr(_http.shutil, 'which', lambda name: '/usr/bin/curl')
    monkeypatch.setattr(_http.subprocess, 'run', fake_run)
    assert _http.curl_download('https://example.invalid/g.nc', out,
                               timeout=1.0, verbose=False)
    assert victim.read_bytes() == b'ORIGINAL'
    assert not out.is_symlink()
    assert out.read_bytes() == b'DOWNLOADED'
