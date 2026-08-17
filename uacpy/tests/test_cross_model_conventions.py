"""Cross-model output conventions, asserted on VALUES, engine by engine.

Pins the three conventions every engine is supposed to share, on one tiny
100-m Pekeris case (fluid half-space bottom, 100 Hz, ranges [0, 500, 1000] m,
depths [50, 110] m — one water-column receiver, one below the seafloor):

1. **r = 0 column is NaN.** A point-source field carries a ``1/sqrt(r)``
   cylindrical-spreading factor that is singular on the source axis, so every
   engine returns the r = 0 column as NaN (no data) and finite values at
   r > 0. RAM regressed here once without any test noticing — the r = 0
   convention was asserted for Kraken/Scooter but never for RAM.

2. **Below-seafloor receivers.** The engines whose solvers stop meshing at
   the seafloor (Bellhop, Scooter, SPARC, RAM) return NaN there — on the
   *requested* depth axis, never a clamped one — while the engines that mesh
   the sediment (Kraken with a penetrable bottom, OAST) return a finite
   physical transmitted field. The suite previously asserted only the
   warnings, not the values.

3. **Field dtype/unit/identity.** Bellhop, Kraken, Scooter and RAM emit
   complex pressure (``unit='Pa'``, ``phase_reference='travelling_wave'``);
   OAST emits real TL in dB (its ``.plt`` carries no phase); SPARC emits real
   ``p(t)`` (``unit='Pa'``, ``phase_reference='time_domain_native'``). RAM's
   PE reference speed is stamped as ``metadata['pe_reference_speed']`` — the
   old ``'c0'`` key is gone.

Each engine runs exactly once (module-scoped fixtures); all three contracts
are asserted on the same Field, so the whole module costs one tiny run per
engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from uacpy import BoundaryProperties, Environment, Receiver, Source
from uacpy.models import RAM, SPARC, Bellhop, Kraken, RunMode, Scooter
from uacpy.models.oases import OAST

pytestmark = pytest.mark.requires_binary

WATER_DEPTH = 100.0
FREQ = 100.0
DEPTHS = np.array([50.0, 110.0])       # in-water, below-seafloor
RANGES = np.array([0.0, 500.0, 1000.0])
I_WATER, I_SUBFLOOR = 0, 1             # rows of DEPTHS


def _pekeris():
    return Environment(
        name='conventions-pekeris', bathymetry=WATER_DEPTH, ssp=1500.0,
        bottom=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.8, attenuation=0.5,
        ),
    )


def _source():
    return Source(depths=50.0, frequencies=FREQ)


def _receiver():
    return Receiver(depths=DEPTHS.copy(), ranges=RANGES.copy())


@pytest.fixture(scope='module')
def bellhop_field():
    return Bellhop(verbose=False).run(
        _pekeris(), _source(), _receiver(), run_mode=RunMode.COHERENT_TL)


@pytest.fixture(scope='module')
def kraken_field():
    return Kraken(verbose=False).run(
        _pekeris(), _source(), _receiver(), run_mode=RunMode.COHERENT_TL)


@pytest.fixture(scope='module')
def scooter_field():
    return Scooter(verbose=False).run(
        _pekeris(), _source(), _receiver(), run_mode=RunMode.COHERENT_TL)


@pytest.fixture(scope='module')
def ram_field():
    return RAM(verbose=False).run(
        _pekeris(), _source(), _receiver(), run_mode=RunMode.COHERENT_TL)


@pytest.fixture(scope='module')
def sparc_field():
    # SPARC's only mode is TIME_SERIES (it synthesises p(t) directly);
    # n_t_out kept small so its per-depth subprocess march stays in seconds.
    return SPARC(verbose=False, n_t_out=256).run(
        _pekeris(), _source(), _receiver(), run_mode=RunMode.TIME_SERIES)


@pytest.fixture(scope='module')
def oast_field():
    return OAST(verbose=False).run(
        _pekeris(), _source(), _receiver(), run_mode=RunMode.COHERENT_TL)


# Every runnable engine; OAST additionally needs the OASES binaries.
ALL_ENGINES = [
    'bellhop_field', 'kraken_field', 'scooter_field', 'ram_field',
    'sparc_field',
    pytest.param('oast_field', marks=pytest.mark.requires_oases),
]
# Engines that mask sub-seafloor receivers to NaN vs. engines that mesh the
# sediment and return the physical transmitted field there (base.py
# _warn_receiver_below_resolvable documents exactly this split).
MASKING_ENGINES = ['bellhop_field', 'scooter_field', 'sparc_field',
                   'ram_field']
TRANSMITTING_ENGINES = [
    'kraken_field',
    pytest.param('oast_field', marks=pytest.mark.requires_oases),
]


def _finite_over_extra_axes(field):
    """(depth, range) boolean map: all samples finite along any extra axis
    (SPARC's time axis), so 2-D and 3-D fields assert identically."""
    data = np.asarray(field.data)
    extra = tuple(range(2, data.ndim))
    return np.isfinite(data).all(axis=extra) if extra else np.isfinite(data)


# ── 1. r = 0 column ───────────────────────────────────────────────────────


@pytest.mark.parametrize('engine', ALL_ENGINES)
def test_r0_column_is_nan_for_every_engine(engine, request):
    """The r = 0 column of a point-source run is NaN at every depth."""
    field = request.getfixturevalue(engine)
    assert float(field.coords['range'][0]) == 0.0
    r0 = np.asarray(field.data)[:, 0, ...]
    assert np.isnan(r0).all(), (
        f"{field.model}: expected the r=0 column to be all-NaN (singular "
        f"1/sqrt(r) point-source spreading), got {r0!r}"
    )


@pytest.mark.parametrize('engine', ALL_ENGINES)
def test_water_column_receiver_is_finite_at_positive_ranges(engine, request):
    """The NaN at r = 0 is a masked column, not a broken run: the in-water
    receiver row is fully finite at every r > 0."""
    field = request.getfixturevalue(engine)
    finite = _finite_over_extra_axes(field)
    assert finite[I_WATER, 1:].all(), (
        f"{field.model}: in-water receiver row has non-finite samples at "
        f"r > 0 (finite map {finite[I_WATER]})"
    )


# ── 2. below-seafloor receivers ───────────────────────────────────────────


@pytest.mark.parametrize('engine', MASKING_ENGINES)
def test_sub_seafloor_receiver_is_nan_for_masking_engines(engine, request):
    """Bellhop/Scooter/SPARC/RAM stop resolving the field at the seafloor:
    the 110-m receiver row (seafloor at 100 m) is all-NaN, at every range."""
    field = request.getfixturevalue(engine)
    row = np.asarray(field.data)[I_SUBFLOOR, ...]
    assert np.isnan(row).all(), (
        f"{field.model}: expected the below-seafloor receiver row to be "
        f"all-NaN, got finite values (the solver clamps or absorbs there — "
        f"any number would misreport where the field was evaluated)"
    )


@pytest.mark.parametrize('engine', TRANSMITTING_ENGINES)
def test_sub_seafloor_receiver_is_finite_for_transmitting_engines(
        engine, request):
    """Kraken (penetrable fluid half-space) and OAST mesh the sediment, so
    the 110-m receiver carries a finite physical transmitted field."""
    field = request.getfixturevalue(engine)
    finite = _finite_over_extra_axes(field)
    assert finite[I_SUBFLOOR, 1:].all(), (
        f"{field.model}: expected a finite transmitted field below the "
        f"seafloor at r > 0, got NaN (finite map {finite[I_SUBFLOOR]})"
    )


@pytest.mark.parametrize('engine', ALL_ENGINES)
def test_requested_depth_axis_is_preserved(engine, request):
    """Masking happens on the requested depth axis — no engine substitutes a
    clamped/native axis for the receiver depths that were asked for."""
    field = request.getfixturevalue(engine)
    np.testing.assert_array_equal(np.asarray(field.coords['depth']), DEPTHS)
    np.testing.assert_array_equal(np.asarray(field.coords['range']), RANGES)


# ── 3. dtype / unit / identity ────────────────────────────────────────────


COMPLEX_PA_ENGINES = ['bellhop_field', 'kraken_field', 'scooter_field',
                      'ram_field']


@pytest.mark.parametrize('engine', COMPLEX_PA_ENGINES)
def test_tl_engines_return_complex_pascal_travelling_wave(engine, request):
    """COHERENT_TL from Bellhop/Kraken/Scooter/RAM is complex pressure in Pa
    with the travelling-wave phase convention; ``.db`` derives real TL."""
    field = request.getfixturevalue(engine)
    assert field.is_complex
    assert field.kind == 'pressure'
    assert field.unit == 'Pa'
    assert field.phase_reference == 'travelling_wave'
    assert list(field.coords) == ['depth', 'range']
    db = np.asarray(field.db)
    assert not np.iscomplexobj(db)
    assert np.isfinite(db[I_WATER, 1:]).all()


@pytest.mark.requires_oases
def test_oast_returns_real_db_tl(oast_field):
    """OAST's .plt carries only real TL, so its Field is real with
    ``unit='dB'`` and ``.db`` is the data itself (no derivation)."""
    assert not oast_field.is_complex
    assert oast_field.kind == 'pressure'
    assert oast_field.unit == 'dB'
    np.testing.assert_array_equal(
        np.asarray(oast_field.db), np.asarray(oast_field.data))


def test_sparc_returns_real_time_domain_pascal(sparc_field):
    """SPARC's p(t) is real linear pressure on a (depth, range, time) grid
    with the time-domain-native phase tag."""
    assert not sparc_field.is_complex
    assert sparc_field.kind == 'pressure'
    assert sparc_field.unit == 'Pa'
    assert sparc_field.phase_reference == 'time_domain_native'
    assert list(sparc_field.coords) == ['depth', 'range', 'time']


@pytest.mark.parametrize('engine', ALL_ENGINES)
def test_result_identity_stamp(engine, request):
    """Every result names its producer and its frequency: ``model`` matches
    the wrapper class, ``backend`` is a concrete binary name, and single-
    frequency runs carry a length-1 ``frequencies`` with ``f0`` the scalar."""
    field = request.getfixturevalue(engine)
    expected_model = {
        'bellhop_field': 'Bellhop', 'kraken_field': 'Kraken',
        'scooter_field': 'Scooter', 'ram_field': 'RAM',
        'sparc_field': 'SPARC', 'oast_field': 'OAST',
    }[engine]
    assert field.model == expected_model
    assert isinstance(field.backend, str) and field.backend
    freqs = np.atleast_1d(np.asarray(field.frequencies, dtype=float))
    assert freqs.shape == (1,)
    assert freqs[0] == FREQ
    assert field.f0 == FREQ


def test_concrete_backend_names(kraken_field, scooter_field, ram_field,
                                sparc_field):
    """The backend stamp names the binary that ran, not the wrapper: Kraken's
    TL comes from field.exe, RAM's flat-bathy dispatch lands on mpiramS."""
    assert kraken_field.backend == 'field'
    assert scooter_field.backend == 'scooter'
    assert ram_field.backend == 'mpiramS'
    assert sparc_field.backend == 'sparc'


def test_ram_metadata_key_is_pe_reference_speed(ram_field):
    """RAM stamps its PE expansion speed as ``pe_reference_speed`` (m/s);
    the ambiguous ``'c0'`` metadata key is retired."""
    md = ram_field.metadata or {}
    assert 'pe_reference_speed' in md, sorted(md)
    assert 'c0' not in md, sorted(md)
    c0 = float(md['pe_reference_speed'])
    assert 1400.0 < c0 < 1800.0   # a sound speed, not a flag or an index
