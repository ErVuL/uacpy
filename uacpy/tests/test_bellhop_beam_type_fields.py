"""Bellhop ``beam_type`` changes the *field*, not just the object.

A per-beam isinstance/isfinite smoke would still pass if the wrapper
silently dropped ``beam_type``. This module pins the full chain on one
small isovelocity case:

* the requested beam letter lands verbatim in the ``.env`` run-type deck
  (``'<run_type><beam_type>…'``, case-significant — ``g`` and ``G`` are
  different beams);
* beam models with different influence routines produce measurably different
  TL (thresholds set ~10x below the measured separations, far above solver
  noise);
* the geometric hat beam gives the same answer in Cartesian (``'G'``) and
  ray-centred (``'g'``) coordinates on this straight-ray isovelocity case —
  the one pair that *should* coincide here;
* the same beam type twice is bit-identical (``backend='fortran'`` is pinned
  for exactly this reason: the CUDA reduction order drifts at ~1e-6 dB, the
  Fortran binary is deterministic).
"""

from __future__ import annotations

import numpy as np
import pytest

from uacpy import Environment, Receiver, Source
from uacpy.models import Bellhop, RunMode

pytestmark = pytest.mark.requires_binary

BEAM_TYPES = ('B', 'G', 'g', 'S', 'C', 'R')


def _case():
    env = Environment(name='beam-field', bathymetry=100.0, ssp=1500.0)
    src = Source(depths=50.0, frequencies=1000.0)
    rcv = Receiver(depths=np.array([25.0, 50.0, 75.0]),
                   ranges=np.linspace(500.0, 3000.0, 6))
    return env, src, rcv


@pytest.fixture(scope='module')
def beam_runs(tmp_path_factory):
    """One COHERENT_TL run per beam type, with the .env deck kept on disk.

    Returns ``{beam_type: (Field, env_deck_text)}``.
    """
    env, src, rcv = _case()
    runs = {}
    for bt in BEAM_TYPES:
        wd = tmp_path_factory.mktemp(f'beam_{bt}_')
        model = Bellhop(verbose=False, beam_type=bt, backend='fortran',
                        work_dir=wd, cleanup=False)
        field = model.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
        deck = next(wd.glob('*.env')).read_text()
        runs[bt] = (field, deck)
    return runs


@pytest.mark.parametrize('beam_type', BEAM_TYPES)
def test_beam_type_reaches_the_env_deck(beam_runs, beam_type):
    """The run-type option line carries the requested beam letter verbatim
    (position 2 of ``'<run_type><beam_type>…'``), case preserved."""
    _, deck = beam_runs[beam_type]
    quoted = [ln.strip().strip("'\"") for ln in deck.splitlines()
              if ln.strip().startswith(("'", '"'))]
    # COHERENT_TL writes run type 'C'; the beam letter follows immediately.
    run_lines = [q for q in quoted
                 if len(q) >= 2 and q[0] == 'C' and q[1] in
                 'BGgSCRbc']
    assert run_lines, f"no run-type option line found in deck:\n{deck}"
    assert run_lines[0][1] == beam_type, (
        f"deck run-type line {run_lines[0]!r} does not carry the requested "
        f"beam_type {beam_type!r}"
    )


def _max_tl_diff(field_a, field_b):
    """max |ΔTL| over the cells finite in both fields (Cerveny beams leave
    out-of-footprint cells as NaN no-data)."""
    tl_a, tl_b = np.asarray(field_a.db), np.asarray(field_b.db)
    both = np.isfinite(tl_a) & np.isfinite(tl_b)
    assert both.any(), "no jointly-finite cells to compare"
    return float(np.abs(tl_a[both] - tl_b[both]).max())


# Measured separations on this case (fortran backend): B-G 0.10 dB,
# B-S 1.24 dB, G-S 1.29 dB, B-C 2.19 dB, B-R 4.07 dB, S-R 4.65 dB.
# Thresholds sit ~5-10x below those, and >1e4x above solver noise (0.0).
DISTINCT_PAIRS = [
    ('B', 'G', 0.02),   # Gaussian vs hat, same Cartesian influence frame
    ('B', 'S', 0.3),    # Gaussian vs simple Gaussian
    ('G', 'S', 0.3),    # hat vs simple Gaussian
    ('B', 'C', 0.5),    # geometric vs Cerveny Cartesian
    ('B', 'R', 0.5),    # geometric vs Cerveny ray-centred
    ('C', 'R', 0.5),    # Cerveny Cartesian vs ray-centred
]


@pytest.mark.parametrize(
    'bt_a,bt_b,min_diff_db',
    DISTINCT_PAIRS,
    ids=[f'{a}-vs-{b}' for a, b, _ in DISTINCT_PAIRS],
)
def test_different_beam_models_give_different_tl(beam_runs, bt_a, bt_b,
                                                 min_diff_db):
    """Beam models with different influence routines must land measurably
    different TL on the same case — the field-level proof that ``beam_type``
    reaches the solver."""
    diff = _max_tl_diff(beam_runs[bt_a][0], beam_runs[bt_b][0])
    assert diff > min_diff_db, (
        f"beam_type {bt_a!r} and {bt_b!r} produced near-identical TL "
        f"(max|dTL|={diff:.4f} dB <= {min_diff_db} dB) — beam_type may not "
        f"be reaching the Bellhop influence routine"
    )


def test_hat_beam_coordinate_systems_agree_on_straight_rays(beam_runs):
    """'G' (Cartesian) and 'g' (ray-centred) are the same hat beam evaluated
    in different coordinates; with straight rays (isovelocity, flat) the two
    influence windows coincide, so jointly-finite cells must agree. A real
    difference here would mean the letters select different physics, not
    different coordinates."""
    diff = _max_tl_diff(beam_runs['G'][0], beam_runs['g'][0])
    assert diff < 0.05, (
        f"hat beams in Cartesian ('G') vs ray-centred ('g') coordinates "
        f"disagree by {diff:.4f} dB on a straight-ray case"
    )


def test_same_beam_type_is_deterministic():
    """Two identical runs are bit-identical (NaN pattern included) on the
    Fortran backend — the control that makes the difference thresholds above
    meaningful."""
    env, src, rcv = _case()
    field_1 = Bellhop(verbose=False, beam_type='B', backend='fortran').run(
        env, src, rcv, run_mode=RunMode.COHERENT_TL)
    field_2 = Bellhop(verbose=False, beam_type='B', backend='fortran').run(
        env, src, rcv, run_mode=RunMode.COHERENT_TL)
    assert np.array_equal(np.asarray(field_1.data), np.asarray(field_2.data),
                          equal_nan=True)
