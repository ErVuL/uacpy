"""Tests for the uniform file-path-in-metadata convention.

Every model writes its primary output paths (and ``prt_file``) into
``result.metadata`` for every run. When ``cleanup=True`` the work dir
is wiped after the wrapper returns, so the recorded paths point at
files that no longer exist — the caller surfaces this naturally as a
``FileNotFoundError`` if they try to open one. When ``cleanup=False``
the paths point at live files.

These tests use representative models that don't require multi-second
binary runs at slow scale.
"""

import os

import numpy as np
import pytest

import uacpy
from uacpy.models import Bellhop, Bounce, Kraken, KrakenField, RAM, SPARC
from uacpy.core import BoundaryProperties


def _basic_setup():
    env = uacpy.Environment(name='t', bathymetry=100.0, ssp=1500.0)
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(
        depths=np.linspace(10, 90, 5),
        ranges=np.linspace(500, 5000, 10),
    )
    return env, src, rcv


def _elastic_env():
    return uacpy.Environment(
        name='elastic', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0, density=1.8, attenuation=0.2,
            shear_speed=400.0, shear_attenuation=0.5,
        ),
    )


# ----------------------------------------------------------------------
# Bellhop
# ----------------------------------------------------------------------

@pytest.mark.requires_binary
def test_bellhop_paths_present_when_cleanup_false(tmp_path):
    env, src, rcv = _basic_setup()
    bh = Bellhop(verbose=False, work_dir=tmp_path)   # cleanup defaults False
    field = bh.run(env, src, rcv)
    assert 'shd_file' in field.metadata
    assert 'prt_file' in field.metadata
    assert os.path.exists(field.metadata['shd_file'])
    assert os.path.exists(field.metadata['prt_file'])


@pytest.mark.requires_binary
def test_bellhop_paths_point_to_missing_files_when_cleanup_true():
    env, src, rcv = _basic_setup()
    bh = Bellhop(verbose=False)                       # default cleanup=True
    field = bh.run(env, src, rcv)
    # Paths are recorded but the files have been wiped.
    assert 'shd_file' in field.metadata
    assert not os.path.exists(field.metadata['shd_file'])


# ----------------------------------------------------------------------
# Kraken (modes solver)
# ----------------------------------------------------------------------

@pytest.mark.requires_binary
def test_kraken_paths_present_when_cleanup_false(tmp_path):
    env, src, rcv = _basic_setup()
    kr = Kraken(verbose=False, work_dir=tmp_path)
    modes = kr.run(env, src, rcv)
    assert 'mod_file' in modes.metadata
    assert os.path.exists(modes.metadata['mod_file'])


@pytest.mark.requires_binary
def test_kraken_paths_point_to_missing_files_when_cleanup_true():
    env, src, rcv = _basic_setup()
    kr = Kraken(verbose=False)
    modes = kr.run(env, src, rcv)
    assert 'mod_file' in modes.metadata
    assert not os.path.exists(modes.metadata['mod_file'])


# ----------------------------------------------------------------------
# Bounce (the original case)
# ----------------------------------------------------------------------

@pytest.mark.requires_binary
def test_bounce_paths_present_when_work_dir_pinned(tmp_path):
    env = _elastic_env()
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
    bn = Bounce(verbose=False, c_low=1400.0, c_high=10000.0, rmax=10000.0,
                work_dir=tmp_path)
    res = bn.run(env, src, rcv)
    assert 'brc_file' in res.metadata
    assert os.path.exists(res.metadata['brc_file'])


@pytest.mark.requires_binary
def test_bounce_paths_point_to_missing_files_when_no_work_dir():
    env = _elastic_env()
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
    bn = Bounce(verbose=False, c_low=1400.0, c_high=10000.0, rmax=10000.0)
    res = bn.run(env, src, rcv)
    # In-memory data still there
    assert res.theta is not None and len(res.theta) > 0
    # Paths recorded but the work dir was wiped.
    assert 'brc_file' in res.metadata
    assert not os.path.exists(res.metadata['brc_file'])


# ----------------------------------------------------------------------
# KrakenField (modes + field pipeline)
# ----------------------------------------------------------------------

@pytest.mark.requires_binary
def test_krakenfield_paths_present_when_cleanup_false(tmp_path):
    env, src, rcv = _basic_setup()
    kf = KrakenField(verbose=False, work_dir=tmp_path)
    field = kf.run(env, src, rcv)
    assert 'shd_file' in field.metadata
    assert os.path.exists(field.metadata['shd_file'])


@pytest.mark.requires_binary
def test_krakenfield_paths_point_to_missing_files_when_cleanup_true():
    env, src, rcv = _basic_setup()
    kf = KrakenField(verbose=False)
    field = kf.run(env, src, rcv)
    assert 'shd_file' in field.metadata
    assert not os.path.exists(field.metadata['shd_file'])


# ----------------------------------------------------------------------
# SPARC (time-domain PE) — slow because each receiver depth spawns a
# binary call. Pass a rigid bottom up front so SPARC's
# 'auto-converting halfspace to rigid' warning doesn't pollute the
# pytest log; the helper's behaviour is what we're testing, not the
# halfspace handling.
# ----------------------------------------------------------------------

def _sparc_env():
    return uacpy.Environment(
        name='sparc-t', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='rigid'),
    )


@pytest.mark.slow
@pytest.mark.requires_binary
def test_sparc_paths_present_when_cleanup_false(tmp_path):
    env = _sparc_env()
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.linspace(500, 3000, 8))
    sp = SPARC(verbose=False, work_dir=tmp_path)
    field = sp.run(env, src, rcv)
    # SPARC writes per-depth/.rts under base_name; one of grn/rts should
    # be picked up by the helper at the wrapper base_name.
    assert (
        'rts_file' in field.metadata
        or 'grn_file' in field.metadata
        or 'prt_file' in field.metadata
    ), (
        f"Expected at least one SPARC output path in metadata; "
        f"got keys: {list(field.metadata)}"
    )


@pytest.mark.slow
@pytest.mark.requires_binary
def test_sparc_paths_point_to_missing_files_when_cleanup_true():
    env = _sparc_env()
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.linspace(500, 3000, 8))
    sp = SPARC(verbose=False)
    field = sp.run(env, src, rcv)
    for key in ('rts_file', 'grn_file', 'prt_file'):
        if key in field.metadata:
            assert not os.path.exists(field.metadata[key])


# ----------------------------------------------------------------------
# RAM (mpiramS backend) — Collins backends use per-call tempdirs and
# never expose persistent paths, so we only test the mpiramS path here.
# ----------------------------------------------------------------------

@pytest.mark.requires_binary
def test_ram_mpirams_paths_present_when_cleanup_false(tmp_path):
    env, src, rcv = _basic_setup()
    ram = RAM(verbose=False, dr=20.0, dz=2.0, work_dir=tmp_path)
    assert ram.select_backend(env) == 'mpiramS'
    field = ram.run(env, src, rcv)
    assert 'psif_file' in field.metadata, (
        f"Expected RAM mpiramS psif_file; got keys: {list(field.metadata)}"
    )
    assert os.path.exists(field.metadata['psif_file'])


@pytest.mark.requires_binary
def test_ram_mpirams_paths_point_to_missing_files_when_cleanup_true():
    env, src, rcv = _basic_setup()
    ram = RAM(verbose=False, dr=20.0, dz=2.0)
    field = ram.run(env, src, rcv)
    if 'psif_file' in field.metadata:
        assert not os.path.exists(field.metadata['psif_file'])


# ----------------------------------------------------------------------
# Cleanup-on-disk: when ``cleanup=True``, the work_dir is actually wiped
# (not just unreferenced from metadata).
# ----------------------------------------------------------------------

@pytest.mark.requires_binary
def test_bellhop_pinned_work_dir_with_cleanup_true_is_wiped(tmp_path):
    """``work_dir`` pinned + ``cleanup=True`` ⇒ directory is removed
    after run(); the recorded paths point at files that no longer exist."""
    env, src, rcv = _basic_setup()
    work = tmp_path / 'bellhop_pinned'
    bh = Bellhop(verbose=False, work_dir=work, cleanup=True)
    field = bh.run(env, src, rcv)
    assert 'shd_file' in field.metadata
    assert not os.path.exists(field.metadata['shd_file'])
    assert not work.exists(), (
        f"cleanup=True should remove the work_dir; {work} still exists"
    )


@pytest.mark.requires_binary
def test_bounce_pinned_work_dir_with_cleanup_true_is_wiped(tmp_path):
    """Bounce: work_dir pinned + cleanup=True ⇒ both .brc/.irc and the
    directory are gone after run()."""
    env = _elastic_env()
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
    work = tmp_path / 'bounce_pinned'
    bn = Bounce(verbose=False, c_low=1400.0, c_high=10000.0, rmax=10000.0,
                work_dir=work, cleanup=True)
    res = bn.run(env, src, rcv)
    # In-memory data still good
    assert res.theta is not None and len(res.theta) > 0
    # Path recorded but the dir is gone.
    assert 'brc_file' in res.metadata
    assert not os.path.exists(res.metadata['brc_file'])
    assert not work.exists()


@pytest.mark.requires_binary
def test_pinned_work_dir_cleanup_false_dir_persists(tmp_path):
    """Negative control: work_dir pinned + cleanup=False (default for
    pinned dir) ⇒ directory survives the call. Sanity-check the dual."""
    env, src, rcv = _basic_setup()
    work = tmp_path / 'persist'
    bh = Bellhop(verbose=False, work_dir=work)   # cleanup defaults False
    bh.run(env, src, rcv)
    assert work.exists() and any(work.iterdir())
