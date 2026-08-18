"""Cross-engine complex-phase contract, referenced to Scooter.

Every engine that emits complex pressure tags it ``travelling_wave`` —
``H(f)`` carries the engineering propagator ``exp(i(wt - kr))`` — so the
same cell of the same environment must come back with the *same phase* from
every engine, not merely the same magnitude. Scooter is the reference: its
in-tree Hankel transform evaluates the wavenumber integral directly, with
the source-geometry factors taken verbatim from ``fieldsco.m``.

The contract is asserted on one tiny Pekeris case per (engine, source
geometry): the amplitude-weighted circular mean of ``arg(engine / Scooter)``
over the receiver grid must sit within ``PHASE_TOL_DEG`` of zero. The
tolerance is generous (±10°) so beam-sum bias (Bellhop, measured ≈ 4.8°) and
parabolic-equation approximations (RAM) pass, while the convention-level
offsets this guards against — a missed line-source ``exp(-i*pi/4)`` (45°), a
sign flip (180°), a conjugated carrier — fail unmistakably.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from uacpy import BoundaryProperties, Environment, Receiver, Source
from uacpy.models import RAM, Bellhop, Kraken, RunMode, Scooter
from uacpy.models.oases import OASP

pytestmark = pytest.mark.requires_binary

WATER_DEPTH = 100.0
FREQ = 100.0
SRC_DEPTH = 25.0
DEPTHS = np.array([20.0, 40.0, 60.0, 80.0])
RANGES = np.linspace(500.0, 3000.0, 6)
PHASE_TOL_DEG = 10.0


def _pekeris():
    return Environment(
        name='phase-pekeris', bathymetry=WATER_DEPTH, ssp=1500.0,
        bottom=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0, density=1.5, attenuation=0.5,
        ),
    )


def _source(source_type):
    return Source(depths=SRC_DEPTH, frequencies=FREQ, source_type=source_type)


def _receiver():
    return Receiver(depths=DEPTHS.copy(), ranges=RANGES.copy())


def _run(model, source_type):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.run(_pekeris(), _source(source_type), _receiver(),
                         run_mode=RunMode.COHERENT_TL)


@pytest.fixture(scope='module')
def scooter_ref():
    """Scooter reference field per source geometry, computed once each."""
    cache = {}

    def get(source_type):
        if source_type not in cache:
            cache[source_type] = _run(Scooter(verbose=False), source_type)
        return cache[source_type]

    return get


def circular_mean_phase_deg(field, ref) -> float:
    """Amplitude-weighted circular mean of ``arg(field / ref)`` in degrees.

    Weighted by ``|ref|`` so interference nulls — where the phase of a ratio
    of two nearly-zero numbers is noise — carry almost no weight.
    """
    k = np.asarray(field.data, dtype=complex).ravel()
    s = np.asarray(ref.data, dtype=complex).ravel()
    ok = (np.isfinite(k) & np.isfinite(s)
          & (np.abs(k) > 0) & (np.abs(s) > 0))
    assert ok.sum() >= 0.5 * k.size, (
        f"too few comparable cells ({int(ok.sum())}/{k.size}) to measure a "
        f"phase offset")
    ratio = k[ok] / s[ok]
    z = np.sum(np.abs(s[ok]) * ratio / np.abs(ratio))
    return float(np.degrees(np.angle(z)))


# (case id, model factory, source geometry). Every engine × source geometry
# it emits complex CW pressure for; the Fortran Bellhop is pinned so the
# assert never rides bellhopcuda's nondeterministic GPU reduction.
CASES = [
    pytest.param(lambda: Kraken(verbose=False), 'point', id='kraken-point'),
    pytest.param(lambda: Kraken(verbose=False), 'line', id='kraken-line'),
    pytest.param(lambda: Kraken(verbose=False), 'scaled', id='kraken-scaled'),
    pytest.param(lambda: Bellhop(verbose=False, backend='fortran'), 'point',
                 id='bellhop-point'),
    pytest.param(lambda: Bellhop(verbose=False, backend='fortran'), 'line',
                 id='bellhop-line'),
    pytest.param(lambda: RAM(verbose=False, backend='mpiramS'), 'point',
                 id='ram-mpiramS-point'),
    pytest.param(lambda: RAM(verbose=False, backend='ramgeo'), 'point',
                 id='ram-collins-point'),
    pytest.param(lambda: OASP(verbose=False), 'point', id='oasp-point',
                 marks=pytest.mark.requires_oases),
]


@pytest.mark.parametrize('factory,source_type', CASES)
def test_phase_agrees_with_scooter(factory, source_type, scooter_ref):
    field = _run(factory(), source_type)
    offset = circular_mean_phase_deg(field, scooter_ref(source_type))
    assert abs(offset) <= PHASE_TOL_DEG, (
        f"{field.model} ({source_type} source): amplitude-weighted mean "
        f"arg(engine/Scooter) = {offset:+.2f} deg exceeds "
        f"±{PHASE_TOL_DEG:g} deg — the complex payloads do not share the "
        f"travelling_wave convention (45° ⇒ a missing line-source "
        f"exp(-i*pi/4); 180° ⇒ a sign flip; sign-mirrored ⇒ a conjugated "
        f"carrier)"
    )


def test_scooter_line_reference_carries_the_2d_phase(scooter_ref):
    """The line-source reference itself differs from the point-source one —
    the two geometries are distinct kernels, so an engine cannot pass the
    line case by ignoring ``source_type``."""
    point = scooter_ref('point')
    line = scooter_ref('line')
    assert not np.allclose(np.asarray(point.data)[np.isfinite(point.data)],
                           np.asarray(line.data)[np.isfinite(line.data)])
