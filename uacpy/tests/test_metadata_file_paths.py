"""Tests for the uniform file-path-in-metadata convention.

Every model writes its primary output paths (and ``prt_file``) into
``result.metadata`` only when the work dir survives the run, i.e. when
``cleanup=False``. With ``cleanup=True`` the work dir is wiped after
the wrapper returns and the ``*_file`` keys are absent from metadata
— the absence is the documented signal (DOCUMENTATION.md §6) that
nothing is on disk to read.

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
def test_bellhop_paths_absent_when_cleanup_true():
    env, src, rcv = _basic_setup()
    bh = Bellhop(verbose=False)                       # default cleanup=True
    field = bh.run(env, src, rcv)
    assert 'shd_file' not in field.metadata
    assert 'prt_file' not in field.metadata


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
def test_kraken_paths_absent_when_cleanup_true():
    env, src, rcv = _basic_setup()
    kr = Kraken(verbose=False)
    modes = kr.run(env, src, rcv)
    assert 'mod_file' not in modes.metadata


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
def test_bounce_paths_absent_when_no_work_dir():
    env = _elastic_env()
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
    bn = Bounce(verbose=False, c_low=1400.0, c_high=10000.0, rmax=10000.0)
    res = bn.run(env, src, rcv)
    assert res.theta is not None and len(res.theta) > 0
    assert 'brc_file' not in res.metadata


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
def test_krakenfield_paths_absent_when_cleanup_true():
    env, src, rcv = _basic_setup()
    kf = KrakenField(verbose=False)
    field = kf.run(env, src, rcv)
    assert 'shd_file' not in field.metadata


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
def test_sparc_paths_absent_when_cleanup_true():
    env = _sparc_env()
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.linspace(500, 3000, 8))
    sp = SPARC(verbose=False)
    field = sp.run(env, src, rcv)
    for key in ('rts_file', 'grn_file', 'prt_file'):
        assert key not in field.metadata


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
def test_ram_mpirams_paths_absent_when_cleanup_true():
    env, src, rcv = _basic_setup()
    ram = RAM(verbose=False, dr=20.0, dz=2.0)
    field = ram.run(env, src, rcv)
    assert 'psif_file' not in field.metadata


# ----------------------------------------------------------------------
# Cleanup-on-disk: when ``cleanup=True``, the work_dir is actually wiped
# (not just unreferenced from metadata).
# ----------------------------------------------------------------------

@pytest.mark.requires_binary
def test_bellhop_pinned_work_dir_with_cleanup_true_is_wiped(tmp_path):
    """``work_dir`` pinned + ``cleanup=True`` ⇒ directory is removed
    after run(); metadata carries no ``*_file`` keys."""
    env, src, rcv = _basic_setup()
    work = tmp_path / 'bellhop_pinned'
    bh = Bellhop(verbose=False, work_dir=work, cleanup=True)
    field = bh.run(env, src, rcv)
    assert 'shd_file' not in field.metadata
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
    assert res.theta is not None and len(res.theta) > 0
    assert 'brc_file' not in res.metadata
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


# ─────────────────────────────────────────────────────────────────────────────
# Documentation-drift detection
# ─────────────────────────────────────────────────────────────────────────────


def _registered_keys_for(model_name: str) -> set:
    """Return the set of metadata keys ``_DOCUMENTED_METADATA`` declares
    for ``model_name``, unioned with the universal keys."""
    from uacpy.core.results import _DOCUMENTED_METADATA, _UNIVERSAL_METADATA
    documented = {k for (m, k) in _DOCUMENTED_METADATA if m == model_name}
    return documented | set(_UNIVERSAL_METADATA)


# (model_cls, run_kwargs) — keep small fast runs; covers the path that
# actually attaches metadata keys. ``work_dir`` is pinned so the
# ``_attach_output_paths`` branch fires (cleanup=True suppresses it).
def _drift_cases():
    from uacpy.models import (
        Bellhop, Bounce, Kraken, KrakenField, RAM, Scooter, SPARC,
        OAST, OASR, OASP,
    )
    return [
        ('Bellhop', Bellhop, {}, {}),
        ('Bounce',  Bounce,  dict(c_low=1400.0, c_high=10000.0, rmax=10000.0), {}),
        ('Kraken',  Kraken,  {}, dict(run_mode=uacpy.RunMode.MODES)),
        ('KrakenField', KrakenField, {}, {}),
        ('RAM',     RAM,     {}, {}),
        ('Scooter', Scooter, {}, {}),
        ('SPARC',   SPARC,   dict(n_t_out=256), {}),
        ('OAST',    OAST,    {}, {}),
        ('OASR',    OASR,    {}, dict(run_mode=uacpy.RunMode.REFLECTION)),
        ('OASP',    OASP,    {}, dict(run_mode=uacpy.RunMode.BROADBAND)),
    ]


@pytest.mark.requires_binary
@pytest.mark.parametrize(
    "name,model_cls,ctor_extras,run_extras",
    _drift_cases(),
    ids=[c[0] for c in _drift_cases()],
)
def test_metadata_keys_are_all_documented(
    name, model_cls, ctor_extras, run_extras, tmp_path,
):
    """Every key written into ``result.metadata`` by a concrete model
    must appear in ``_DOCUMENTED_METADATA`` (or ``_UNIVERSAL_METADATA``).

    Catches drift: a wrapper adding a new ``result.metadata[k] = v`` line
    without registering ``(model_name, k)`` lets ``Result.list_metadata()``
    silently report the new key as undocumented. This parametrized check
    fails loudly instead.
    """
    env, src, rcv = _basic_setup()
    if name == 'OASR':
        # OASR needs an elastic bottom for a meaningful reflection result.
        env = _elastic_env()
    work = tmp_path / f'{name.lower()}_drift'
    model = model_cls(work_dir=work, cleanup=False, verbose=False, **ctor_extras)
    result = model.run(env, src, rcv, **run_extras)

    attached = set(result.metadata.keys())
    documented = _registered_keys_for(result.model)
    undocumented = attached - documented
    assert not undocumented, (
        f"{result.model}: result.metadata has key(s) {sorted(undocumented)} "
        f"that are not registered in _DOCUMENTED_METADATA. "
        f"Add an entry per (model, key) in uacpy/core/results.py or fix the "
        f"wrapper to drop the unregistered key."
    )
