"""Cross-model agreement on canonical generic problems.

Each ``Scenario`` defines an environment + source + receiver and lists the
models that should produce a sufficiently-similar TL field. One reference
model is named per scenario; every other applicable model is compared
against it pairwise with an RMSE tolerance computed in a stable range
window (avoiding the very near field where geometric singularities and
self-starter artefacts dominate).

Adding a scenario:
1. Append a ``Scenario(name=..., env=..., source=..., receiver=...,
   reference=..., comparisons=[...], tolerance_db=...)`` to ``SCENARIOS``.
2. Each entry in ``comparisons`` is a ``(label, callable)`` pair, where
   the callable takes ``(env, source, receiver)`` and returns a
   ``Field`` with ``field_type='tl'`` (uacpy's standard TL field).

Tests are parametrised over ``(scenario, comparison)`` so the failure
report tells you exactly which model disagreed on which scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np
import pytest

import uacpy
from uacpy.core.environment import (
    BoundaryProperties, Environment, LayeredBottom, SedimentLayer,
)
from uacpy.core.receiver import Receiver
from uacpy.core.source import Source
from uacpy.models import (
    Bellhop, Kraken, KrakenField, RAM, RunMode, Scooter,
)

# Detect availability of compiled RAM-family binaries — env-dependent so
# the rough-surface scenario can be skipped cleanly when ramsurf isn't built.
def _has_ramsurf() -> bool:
    try:
        ram = RAM(verbose=False)
        ram._collins_binary('ramsurf')
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.requires_binary


# ─── Scenario plumbing ─────────────────────────────────────────────────────


@dataclass
class Scenario:
    """One generic problem with a reference model and a list of models that
    should agree with it.

    ``comparisons`` is a list of ``(label, runner)`` or ``(label, runner,
    tolerance_db)`` tuples. The third element overrides the scenario-level
    ``tolerance_db`` for that one comparison — useful when ray-vs-mode or
    PE-vs-mode physics disagree more than mode-vs-mode but you still want
    to track the agreement.
    """
    name: str
    env: Environment
    source: Source
    receiver: Receiver
    reference_label: str
    reference: Callable[[Environment, Source, Receiver], 'uacpy.core.results.Result']
    comparisons: List[Tuple] = field(default_factory=list)
    tolerance_db: float = 3.0
    range_window_m: Tuple[float, float] = (1000.0, 8000.0)


# ─── Reference / comparison runners ───────────────────────────────────────


def _kraken_field_tl(env, src, rcv):
    """KrakenField.run → COHERENT_TL Field."""
    return KrakenField(verbose=False).run(env, src, rcv, run_mode=RunMode.COHERENT_TL)


def _scooter_tl(env, src, rcv):
    return Scooter(verbose=False).run(env, src, rcv, run_mode=RunMode.COHERENT_TL)


def _bellhop_tl(env, src, rcv):
    return Bellhop(verbose=False).run(env, src, rcv, run_mode=RunMode.COHERENT_TL)


def _ram_tl(env, src, rcv):
    """RAM dispatcher — picks mpiramS / rams / ramsurf based on env."""
    return RAM(verbose=False).run(env, src, rcv, run_mode=RunMode.COHERENT_TL)


# ─── Scenarios ─────────────────────────────────────────────────────────────


def _pekeris_fluid() -> Scenario:
    env = Environment(
        name='pekeris-fluid', bathymetry=100.0, sound_speed=1500.0,
        bottom=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.7, attenuation=0.5,
        ),
    )
    src = Source(depths=36.0, frequencies=50.0)
    rcv = Receiver(
        depths=np.array([36.0]),
        ranges=np.linspace(200.0, 8000.0, 50),
    )
    return Scenario(
        name='pekeris-fluid-50Hz-36m',
        env=env, source=src, receiver=rcv,
        reference_label='KrakenField',
        reference=_kraken_field_tl,
        comparisons=[
            # Mode-vs-mode and PE-vs-mode agreement is tight on Pekeris.
            ('Scooter', _scooter_tl, 3.0),
            # mpiramS dz floor caps RMSE ~3.8 dB; pin dr/dz in _ram_tl if tighter.
            ('RAM(mpiramS)', _ram_tl, 4.0),
            # Bellhop rays vs Kraken modes is naturally looser at low
            # frequency / few modes — empirically ~5 dB RMSE on this case.
            ('Bellhop', _bellhop_tl, 6.0),
        ],
        tolerance_db=3.0,
    )


def _pekeris_elastic() -> Scenario:
    """The Pekeris-elastic scenario — the canonical RAMS-vs-KrakenC validation.

    Tuning rationale: ``RAM(...)`` defaults to ``np_pade=6`` and
    ``rams_theta=45`` (see uacpy/models/ram.py:_run_collins). On this
    scenario the dispatcher routes to rams0.5; with those defaults the
    RMSE against KrakenField (which auto-routes to KrakenC for elastic)
    is ~1.5 dB over the 1-8 km window and TL @ 5 km matches within 0.1 dB.
    """
    elastic_layered = LayeredBottom(
        layers=[SedimentLayer(
            thickness=10.0, sound_speed=1700.0, density=1.8,
            attenuation=0.2, shear_speed=400.0, shear_attenuation=0.5,
        )],
        halfspace=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.8, attenuation=0.2,
            shear_speed=400.0, shear_attenuation=0.5,
        ),
    )
    elastic_halfspace = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0, shear_speed=400.0, density=1.8,
        attenuation=0.2, shear_attenuation=0.5,
    )
    env_layered = Environment(
        name='pekeris-elastic-layered', bathymetry=100.0, sound_speed=1500.0, bottom=elastic_layered,
    )
    env_halfspace = Environment(
        name='pekeris-elastic-halfspace', bathymetry=100.0, sound_speed=1500.0, bottom=elastic_halfspace,
    )
    src = Source(depths=36.0, frequencies=50.0)
    rcv = Receiver(
        depths=np.array([36.0]),
        ranges=np.linspace(200.0, 8000.0, 50),
    )

    # KrakenField wants the half-space form (elastic Comp selector applies
    # to a single halfspace). RAMS wants the layered form (the writer
    # emits the layered bottom as a Collins piecewise profile). The
    # underlying physics is the same; both bottoms describe the same
    # elastic medium below the seafloor.
    def reference(env_unused, src, rcv):
        return _kraken_field_tl(env_halfspace, src, rcv)

    # RAMS is sensitive to (dr, dz, np_pade, theta); the values below
    # were tuned against the KrakenC reference (RMSE ≈ 1.5 dB over
    # 1-8 km, TL@5km within 0.1 dB). The README of the upstream code
    # explicitly notes that RAMS needs hand-tuning per problem; uacpy's
    # default ``rams_theta=45`` and ``np_pade=6`` come from this scenario.
    def rams(env_unused, src, rcv):
        ram = RAM(verbose=False, np_pade=6, dr=2.0, dz=0.25, zmax=400.0,
                  rams_theta=45.0)
        return ram.run(env_layered, src, rcv, run_mode=RunMode.COHERENT_TL)

    return Scenario(
        name='pekeris-elastic-50Hz-36m',
        env=env_layered, source=src, receiver=rcv,
        reference_label='KrakenField (auto-KrakenC)',
        reference=reference,
        comparisons=[('RAM(rams0.5)', rams, 3.0)],
        tolerance_db=3.0,
    )


def _altimetry_consistency() -> Scenario:
    """Bellhop and RAM(ramsurf) should describe the *same* surface, not the
    inverse of each other. ``env.altimetry`` follows uacpy's "positive up"
    convention; the RAM dispatcher converts to ramsurf's "depth below z=0"
    convention internally. This scenario guards against any sign drift.

    Loose tolerance (8 dB RMSE) because ray-vs-PE on a rough-surface
    Pekeris waveguide is genuinely a different physics, but it's enough to
    catch a sign flip — an inverted-surface scenario shows >25 dB RMSE.
    Range window kept short to stay in the regime where rays converge
    without too much surface-loss accumulation.
    """
    surface = [
        (0.0, 0.0),
        (1500.0, -1.5),  # ice-keel-style depression
        (3000.0, 0.0),
        (4500.0, -1.5),
        (6000.0, 0.0),
    ]
    fluid = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0, density=1.7, attenuation=0.5,
    )
    env = Environment(
        name='altimetry-rough', bathymetry=100.0, sound_speed=1500.0, bottom=fluid, altimetry=surface,
    )
    src = Source(depths=50.0, frequencies=200.0)
    rcv = Receiver(
        depths=np.array([50.0]),
        ranges=np.linspace(500.0, 6000.0, 30),
    )
    return Scenario(
        name='altimetry-consistency-bellhop-vs-ramsurf',
        env=env, source=src, receiver=rcv,
        reference_label='Bellhop',
        reference=lambda env, s, r: Bellhop(verbose=False).run(
            env, s, r, run_mode=RunMode.COHERENT_TL),
        comparisons=[
            # Ray-vs-PE on a rough Pekeris surface naturally diverges past
            # ~3 km as surface multipaths accumulate; 8 dB RMSE is the
            # empirical bar in 1-5 km. The test still catches sign flips
            # cleanly — an inverted surface produces RMSE > 25 dB.
            ('RAM(ramsurf1.5)', lambda env, s, r: RAM(verbose=False).run(
                env, s, r, run_mode=RunMode.COHERENT_TL), 8.0),
        ],
        tolerance_db=8.0,
        range_window_m=(1000.0, 5000.0),
    )


def _pekeris_fluid_hf() -> Scenario:
    """Higher-frequency Pekeris (250 Hz). With ~20 modes the ray-mode
    agreement tightens — Bellhop, Scooter, and RAM(mpiramS) should all
    track KrakenField within a few dB. A second test point above the
    50 Hz scenario gives the framework a frequency-dependence handle.
    """
    env = Environment(
        name='pekeris-fluid-hf', bathymetry=100.0, sound_speed=1500.0,
        bottom=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.7, attenuation=0.5,
        ),
    )
    src = Source(depths=36.0, frequencies=250.0)
    rcv = Receiver(
        depths=np.array([36.0]),
        ranges=np.linspace(200.0, 8000.0, 50),
    )
    return Scenario(
        name='pekeris-fluid-250Hz-36m',
        env=env, source=src, receiver=rcv,
        reference_label='KrakenField',
        reference=_kraken_field_tl,
        comparisons=[
            ('Scooter', _scooter_tl, 4.0),
            ('RAM(mpiramS)', _ram_tl, 4.0),
            ('Bellhop', _bellhop_tl, 6.0),
        ],
        tolerance_db=4.0,
    )


def _pekeris_elastic_broadband_at_fc() -> Scenario:
    """RAMS broadband validation against KrakenField broadband.

    Same Pekeris-elastic env as ``_pekeris_elastic`` but exercises the
    full BROADBAND path: ``rams0.5`` is driven in a Python frequency
    loop reading the patched ``pcomplex.bin``, yielding an engineering
    travelling-wave H(f). The agreement is checked on the TL slice at
    the centre frequency — that's where ``rams_theta`` has been tuned
    and where KrakenField's modal sum is best resolved. Per-frequency
    RMSE across the full band is naturally looser (~5 dB) due to RAMS'
    theta sensitivity vs. frequency; the centre-frequency agreement is
    the meaningful regression anchor.
    """
    elastic_layered = LayeredBottom(
        layers=[SedimentLayer(
            thickness=10.0, sound_speed=1700.0, density=1.8,
            attenuation=0.2, shear_speed=400.0, shear_attenuation=0.5,
        )],
        halfspace=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.8, attenuation=0.2,
            shear_speed=400.0, shear_attenuation=0.5,
        ),
    )
    elastic_halfspace = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0, shear_speed=400.0, density=1.8,
        attenuation=0.2, shear_attenuation=0.5,
    )
    env_layered = Environment(
        name='pekeris-elastic-bb-layered', bathymetry=100.0, sound_speed=1500.0, bottom=elastic_layered,
    )
    env_halfspace = Environment(
        name='pekeris-elastic-bb-halfspace', bathymetry=100.0, sound_speed=1500.0, bottom=elastic_halfspace,
    )
    src = Source(depths=36.0, frequencies=50.0)
    rcv = Receiver(
        depths=np.array([36.0]),
        ranges=np.linspace(500.0, 8000.0, 30),
    )

    def _bb_to_fc_tl(field):
        # Pick the centre-frequency slice from a transfer-function field
        # and convert |H(fc, ·)| to dB for comparison.
        i_fc = int(np.argmin(np.abs(field.frequencies - 50.0)))
        return -20.0 * np.log10(np.abs(field.data[:, :, i_fc]) + 1e-20)

    def reference(env_unused, src_, rcv_):
        kf = KrakenField(verbose=False).run(
            env_halfspace, src_, rcv_,
            frequencies=np.linspace(25.5, 74.5, 99),
            run_mode=RunMode.BROADBAND,
        )
        # Inject the centre-freq TL into a fake 'tl' Field-like object the
        # comparator already understands. Build an ad-hoc Field with TL only.
        from uacpy.core.results import TLField
        return TLField(
            data=_bb_to_fc_tl(kf),
            depths=kf.depths, ranges=kf.ranges,
            model='KrakenField', frequencies=50.0,
        )

    def rams_bb(env_unused, src_, rcv_):
        ram = RAM(verbose=False, np_pade=6, dr=2.0, dz=0.25, zmax=400.0,
                  rams_theta=45.0, Q=2.0, T=2.0)
        hf = ram.run(env_layered, src_, rcv_, run_mode=RunMode.BROADBAND)
        from uacpy.core.results import TLField
        return TLField(
            data=_bb_to_fc_tl(hf),
            depths=hf.depths, ranges=hf.ranges,
            model='RAM(rams)', frequencies=50.0,
        )

    return Scenario(
        name='pekeris-elastic-broadband-50Hz-fc-slice',
        env=env_layered, source=src, receiver=rcv,
        reference_label='KrakenField broadband (fc slice)',
        reference=reference,
        comparisons=[('RAM(rams0.5) broadband', rams_bb, 4.0)],
        tolerance_db=4.0,
        range_window_m=(1000.0, 7000.0),
    )


def _altimetry_broadband_at_fc() -> Scenario:
    """ramsurf BROADBAND validation against Bellhop on the same env.

    Same Pekeris+altimetry env as ``_altimetry_consistency`` but
    exercises the full BROADBAND path: ``ramsurf1.5`` is driven in a
    Python frequency loop reading the patched ``pcomplex.bin``,
    yielding an engineering travelling-wave H(f). The agreement is
    checked on the centre-frequency TL slice; full broadband ray-vs-PE
    on a rough surface drifts more across the band so the fc slice is
    the meaningful regression anchor.
    """
    fluid = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0, density=1.7, attenuation=0.5,
    )
    surface = [
        (0.0, 0.0),
        (1500.0, -1.5),
        (3000.0, 0.0),
        (4500.0, -1.5),
        (6000.0, 0.0),
    ]
    env = Environment(
        name='altimetry-rough-bb', bathymetry=100.0, sound_speed=1500.0, bottom=fluid, altimetry=surface,
    )
    src = Source(depths=50.0, frequencies=200.0)
    rcv = Receiver(
        depths=np.array([50.0]),
        ranges=np.linspace(500.0, 6000.0, 30),
    )

    def _bb_to_fc_tl(field):
        i_fc = int(np.argmin(np.abs(field.frequencies - 200.0)))
        return -20.0 * np.log10(np.abs(field.data[:, :, i_fc]) + 1e-20)

    def reference(env_, src_, rcv_):
        bh = Bellhop(verbose=False).run(
            env_, src_, rcv_, run_mode=RunMode.COHERENT_TL,
        )
        return bh

    def ramsurf_bb(env_, src_, rcv_):
        ram = RAM(verbose=False, np_pade=6, dr=2.0, dz=0.25, zmax=400.0,
                  Q=2.0, T=2.0)
        hf = ram.run(env_, src_, rcv_, run_mode=RunMode.BROADBAND)
        from uacpy.core.results import TLField
        return TLField(
            data=_bb_to_fc_tl(hf),
            depths=hf.depths, ranges=hf.ranges,
            model='RAM(ramsurf)', frequencies=200.0,
        )

    return Scenario(
        name='altimetry-broadband-200Hz-fc-slice',
        env=env, source=src, receiver=rcv,
        reference_label='Bellhop',
        reference=reference,
        # Rough-surface ray/PE phase drift dominates; ~9 dB RMSE empirical.
        comparisons=[('RAM(ramsurf1.5) broadband', ramsurf_bb, 9.0)],
        tolerance_db=9.0,
        range_window_m=(1000.0, 5000.0),
    )


SCENARIOS: List[Scenario] = [
    _pekeris_fluid(),
    _pekeris_fluid_hf(),
    _pekeris_elastic(),
    _pekeris_elastic_broadband_at_fc(),
]
if _has_ramsurf():
    SCENARIOS.append(_altimetry_broadband_at_fc())
if _has_ramsurf():
    SCENARIOS.append(_altimetry_consistency())


def _comparison_pairs():
    """Flatten (scenario, label, callable, tolerance) for parametrize."""
    out = []
    for s in SCENARIOS:
        for entry in s.comparisons:
            if len(entry) == 2:
                label, fn = entry
                tol = s.tolerance_db
            elif len(entry) == 3:
                label, fn, tol = entry
            else:
                raise ValueError(
                    f"Scenario {s.name!r}: comparison entry must be "
                    f"(label, fn) or (label, fn, tolerance_db); got {entry!r}"
                )
            out.append(pytest.param(s, label, fn, tol, id=f"{s.name}::{label}"))
    return out


# ─── The agreement check ──────────────────────────────────────────────────


def _rmse_in_window(tl_a: np.ndarray, tl_b: np.ndarray, ranges: np.ndarray,
                    window: Tuple[float, float]) -> Tuple[float, float]:
    rmin, rmax = window
    mask = (
        (ranges >= rmin) & (ranges <= rmax)
        & np.isfinite(tl_a) & np.isfinite(tl_b)
    )
    diff = tl_a[mask] - tl_b[mask]
    return float(np.sqrt(np.mean(diff ** 2))), float(np.max(np.abs(diff)))


@pytest.mark.parametrize("scenario,label,callable_,tolerance", _comparison_pairs())
def test_cross_model_agreement(scenario: Scenario, label: str, callable_,
                               tolerance: float):
    """For each (scenario, comparison) pair, assert RMSE within tolerance.

    Reports both RMSE and max|err| even on success so the test output
    builds up an empirical agreement table over time.
    """
    ref_field = scenario.reference(scenario.env, scenario.source, scenario.receiver)
    cmp_field = callable_(scenario.env, scenario.source, scenario.receiver)

    # Pick the receiver-depth and ranges shared by both (single-depth
    # scenarios are the simple case; for multi-depth, take depth 0).
    ref_tl = np.asarray(ref_field.data).squeeze()
    cmp_tl = np.asarray(cmp_field.data).squeeze()
    if ref_tl.ndim == 2:
        ref_tl = ref_tl[0]
    if cmp_tl.ndim == 2:
        cmp_tl = cmp_tl[0]

    ranges = np.asarray(scenario.receiver.ranges)
    rmse, mxe = _rmse_in_window(cmp_tl, ref_tl, ranges, scenario.range_window_m)

    assert rmse <= tolerance, (
        f"{scenario.name} :: {label} disagrees with "
        f"{scenario.reference_label}: RMSE={rmse:.2f} dB > {tolerance} dB "
        f"(max|err|={mxe:.2f} dB) in window {scenario.range_window_m}"
    )
