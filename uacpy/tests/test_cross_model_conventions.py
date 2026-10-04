"""Cross-model output conventions, asserted on VALUES, engine by engine.

Pins the conventions the engines share on one tiny 100-m Pekeris case (fluid
half-space bottom, 100 Hz, ranges [0, 500, 1000] m, depths [50, 110] m — one
water-column receiver, one below the seafloor). What every registered engine
owes on the same grid (the NaN r = 0 column and its warning, a finite water
column at r > 0, the requested axes kept, the ``PhaseReference`` member) is
held by ``test_engine_conformance.py``; this module pins which engine does
what:

1. **Below-seafloor receivers.** The engines whose solvers stop meshing at
   the seafloor (Bellhop, Scooter, SPARC, RAM) return NaN there — on the
   *requested* depth axis, never a clamped one — while the engines that mesh
   the sediment (Kraken with a penetrable bottom, OAST) return a finite
   physical transmitted field. The suite previously asserted only the
   warnings, not the values.

2. **Field dtype/unit/identity.** Bellhop, Kraken, Scooter and RAM emit
   complex pressure (``unit='Pa'``, ``phase_reference='travelling_wave'``);
   OAST emits real TL in dB (its ``.plt`` carries no phase); SPARC emits real
   ``p(t)`` (``unit='Pa'``, ``phase_reference='time_domain_native'``). RAM's
   PE reference speed is the run settings' ``c0``.

Each engine runs exactly once (module-scoped fixtures); both contracts are
asserted on the same Field, so the whole module costs one tiny run per
engine.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import uacpy
from uacpy import BoundaryProperties, Environment, Receiver, Source
from uacpy.models import RAM, SPARC, Bellhop, Kraken, RunMode, Scooter
from uacpy.models.oases import OASP, OAST
from uacpy.tests.conftest import make_pekeris

pytestmark = pytest.mark.requires_binary

WATER_DEPTH = 100.0
FREQ = 100.0
DEPTHS = np.array([50.0, 110.0])       # in-water, below-seafloor
RANGES = np.array([0.0, 500.0, 1000.0])
I_WATER, I_SUBFLOOR = 0, 1             # rows of DEPTHS


def _pekeris(roughness=0.0):
    return make_pekeris(name='conventions-pekeris', bathymetry=WATER_DEPTH,
                        roughness=roughness)


def _source():
    return Source(depths=50.0, frequencies=FREQ)


def _receiver():
    return Receiver(depths=DEPTHS.copy(), ranges=RANGES.copy())


def _run_quietly(runner):
    """``runner()``, with the warnings its model run raises silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return runner()


@pytest.fixture(scope='module')
def bellhop_field():
    return _run_quietly(lambda: Bellhop(verbose=False).run(
        _pekeris(), _source(), _receiver(), run_mode=RunMode.COHERENT_TL))


@pytest.fixture(scope='module')
def kraken_field():
    return _run_quietly(lambda: Kraken(verbose=False).run(
        _pekeris(), _source(), _receiver(), run_mode=RunMode.COHERENT_TL))


@pytest.fixture(scope='module')
def scooter_field():
    return _run_quietly(lambda: Scooter(verbose=False).run(
        _pekeris(), _source(), _receiver(), run_mode=RunMode.COHERENT_TL))


@pytest.fixture(scope='module')
def ram_field():
    return _run_quietly(lambda: RAM(verbose=False).run(
        _pekeris(), _source(), _receiver(), run_mode=RunMode.COHERENT_TL))


@pytest.fixture(scope='module')
def sparc_field():
    # SPARC's only mode is TIME_SERIES (it synthesises p(t) directly);
    # n_time_samples kept small so its per-depth subprocess march stays in seconds.
    # SPARC refuses a half-space bottom, so it marches the same water over a
    # rigid floor, chosen explicitly.
    env = _pekeris()
    rigid = Environment(name='conventions-rigid-floor', bathymetry=env.depth,
                        ssp=env.ssp,
                        bottom=BoundaryProperties(acoustic_type='rigid'))
    return _run_quietly(lambda: SPARC(verbose=False, n_time_samples=256).run(
        rigid, _source(), _receiver(), run_mode=RunMode.TIME_SERIES))


@pytest.fixture(scope='module')
def oast_field():
    return _run_quietly(lambda: OAST(verbose=False).run(
        _pekeris(), _source(), _receiver(), run_mode=RunMode.COHERENT_TL))


@pytest.fixture(scope='module')
def oasp_field():
    return _run_quietly(lambda: OASP(verbose=False).run(
        _pekeris(), _source(), _receiver(), run_mode=RunMode.COHERENT_TL))


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


# ── 1. below-seafloor receivers ───────────────────────────────────────────


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


# ── 2. dtype / unit / identity ────────────────────────────────────────────


COMPLEX_PA_ENGINES = ['bellhop_field', 'kraken_field', 'scooter_field',
                      'ram_field']


@pytest.mark.parametrize('engine', COMPLEX_PA_ENGINES)
def test_tl_engines_return_complex_pascal_travelling_wave(engine, request):
    """COHERENT_TL from Bellhop/Kraken/Scooter/RAM is complex pressure in Pa
    with the travelling-wave phase convention; ``.dB`` derives real TL."""
    field = request.getfixturevalue(engine)
    assert field.is_complex
    assert field.kind == 'pressure'
    assert field.unit == 'Pa'
    assert field.phase_reference == 'travelling_wave'
    assert list(field.coords) == ['depth', 'range']
    dB = np.asarray(field.dB)
    assert not np.iscomplexobj(dB)
    assert np.isfinite(dB[I_WATER, 1:]).all()


def test_no_model_bypasses_the_phase_reference_coercion():
    """The writer-side half: in ``uacpy/models/``, a literal
    ``phase_reference=`` may only be an argument to ``_result_kwargs`` or
    ``_stamp_result`` (or the ``_extract`` functions they call,
    ``result_kwargs`` / ``stamp_result``), which coerce it. Passing it straight to a results
    constructor is what produces the bare string
    ``test_engine_conformance.py::test_every_mode_returns_its_declared_output``
    catches, and it is invisible to any single-engine assertion."""
    import ast
    from pathlib import Path
    from uacpy.tests._doc_gate import (engines_missing_from,
                                       package_python_files)
    models_dir = Path(uacpy.__file__).parent / 'models'
    paths = package_python_files(models_dir)
    missing = engines_missing_from(paths, models_dir)
    assert not missing, f"the sweep read no file of the engine(s) {missing}"
    offenders = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            literal = any(
                kw.arg == 'phase_reference'
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
                for kw in node.keywords)
            if not literal:
                continue
            callee = node.func
            name = (callee.attr if isinstance(callee, ast.Attribute)
                    else getattr(callee, 'id', ''))
            if name not in ('_result_kwargs', '_stamp_result',
                            'result_kwargs', 'stamp_result'):
                offenders.append(f"{path.relative_to(models_dir)}:"
                                 f"{node.lineno} -> {name}(...)")
    assert not offenders, (
        "a bare-string phase_reference reaches a results constructor "
        "uncoerced at: " + ", ".join(offenders))


@pytest.mark.requires_oases
def test_oasp_shares_the_travelling_wave_sign_with_scooter(oasp_field):
    """OASP's complex pressure is the same travelling-wave convention the
    reference engines carry — the ratio to Scooter is +1, not -1.

    Scooter runs at OASP's own FFT-bin frequency (the ladder rarely lands a
    bin exactly on the request), so the comparison isolates the sign from
    the bin-substitution phase drift. A convention flip would put the
    angles near 180 deg."""
    assert oasp_field.phase_reference == 'travelling_wave'
    assert oasp_field.is_complex
    f_bin = float(np.atleast_1d(oasp_field.frequencies)[0])
    sco = Scooter(verbose=False).run(
        _pekeris(), Source(depths=50.0, frequencies=f_bin), _receiver(),
        run_mode=RunMode.COHERENT_TL)
    ratio = (np.asarray(oasp_field.data)[I_WATER, 1:]
             / np.asarray(sco.data)[I_WATER, 1:])
    angles = np.abs(np.angle(ratio, deg=True))
    assert (angles < 45.0).all(), (
        f"arg(OASP/Scooter) = {np.angle(ratio, deg=True)} deg — a sign "
        f"flip reads ~180 deg"
    )
    assert np.abs(ratio) == pytest.approx(1.0, rel=0.15)


@pytest.mark.requires_oases
def test_oasn_replicas_are_scooters_pressure_on_the_same_geometry():
    """OASES-2 / OASES-3: an OASN replica is the complex pressure a unit
    source at the candidate point puts on the array, so on the same geometry
    it is Scooter's field, sign included.

    OASN writes the normal stress ``σ_zz = -p``; the wrapper negates it and
    tags the travelling-wave convention, so the array-summed phase to
    Scooter is ~0 deg (the raw stress reads ~180). With a pinned wavenumber
    count large enough for the 5 km grid the level converges onto Scooter:
    measured 0.00-0.02 dB at 2-5 km for NW = 16384 and 0.02-0.10 dB for
    8192, where the automatic default reads 0.5-3.3 dB low toward the
    grid's far edge."""
    from uacpy.models import OASN
    arr = np.array([20.0, 40.0, 60.0, 80.0])
    xs = np.array([2000.0, 3000.0, 4000.0, 5000.0])
    env = make_pekeris(name='replica-pekeris', bathymetry=100.0)
    src = Source(depths=30.0, frequencies=150.0)
    sco = Scooter(verbose=False).run(env, src, Receiver(depths=arr,
                                                        ranges=xs))
    P = np.asarray(sco.data).T                                # (x, rcv)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rep = OASN(verbose=False, replica_xmin=1000.0, replica_xmax=5000.0, replica_nx=5,
                   replica_zmin=30.0, replica_zmax=30.0, replica_nz=1, n_wavenumbers=16384).run(
            env, src, Receiver(depths=arr, ranges=[0.0]), RunMode.REPLICA)
    assert rep.phase_reference == 'travelling_wave'
    np.testing.assert_allclose(rep.candidates['x'][1:], xs)
    R = np.asarray(rep.replicas)[0, 0, 1:, 0, :]              # (x, rcv)
    phase = np.angle(np.sum(R * np.conj(P), axis=1), deg=True)
    assert (np.abs(phase) < 5.0).all(), phase
    level = 10 * np.log10(np.sum(np.abs(R) ** 2, axis=1)
                          / np.sum(np.abs(P) ** 2, axis=1))
    assert (np.abs(level) < 0.2).all(), level


@pytest.mark.requires_oases
def test_oast_line_source_matches_scooters_line_source_tl():
    """OASES-11: ``Source(source_type='line')`` runs OAST's plane geometry
    (option 'P', written from the Source) and its TL is the package's
    line-source level — Scooter's on the same geometry, measured within
    0.35 dB from 1 km out. A point source on the same grid is ~30 dB
    weaker, so the tolerance cannot pass on the wrong geometry."""
    env = make_pekeris(name='line-pekeris', bathymetry=100.0)
    rcv = Receiver(depths=[50.0], ranges=np.linspace(1000.0, 5000.0, 9))
    line = Source(depths=25.0, frequencies=100.0, source_type='line')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        oast = OAST(verbose=False).run(env, line, rcv)
    assert 'P' in oast.run_settings.engine.options.split()
    sco = Scooter(verbose=False).run(env, line, rcv)
    diff = np.asarray(oast.data)[0] - np.asarray(sco.dB)[0]
    assert np.abs(diff).max() < 0.75, diff


@pytest.mark.requires_oases
def test_oasp_line_source_is_scooters_line_source_pressure():
    """Decision A4: ``Source(source_type='line')`` runs OASP's plane
    geometry (option 'P', written from the Source), and its complex pressure
    is Scooter's line-source pressure on the same geometry: |ratio| within
    0.1 and phase within 10 deg from 1 km out. A point source on the same
    grid is ~30 dB weaker, so the ratio cannot pass on the wrong
    geometry."""
    env = make_pekeris(name='line-pekeris', bathymetry=100.0)
    rcv = Receiver(depths=[50.0], ranges=np.linspace(1000.0, 5000.0, 9))
    line = Source(depths=25.0, frequencies=100.0, source_type='line')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        oasp = OASP(verbose=False).run(env, line, rcv)
    assert 'P' in oasp.run_settings.engine.options.split()
    f_bin = float(np.atleast_1d(oasp.frequencies)[0])
    sco = Scooter(verbose=False).run(
        env, Source(depths=25.0, frequencies=f_bin, source_type='line'), rcv)
    ratio = np.asarray(oasp.data)[0] / np.asarray(sco.data)[0]
    assert np.abs(np.abs(ratio) - 1.0).max() < 0.1, np.abs(ratio)
    assert np.abs(np.angle(ratio, deg=True)).max() < 10.0, \
        np.angle(ratio, deg=True)


@pytest.mark.requires_oases
def test_oasp_coherent_tl_is_the_field_at_the_source_frequency():
    """OASES-1: ``OASP().run(env, Source(frequencies=100), rcv)`` returns
    the field at 100 Hz — its phase to Scooter at 100 Hz stays within a few
    degrees out to 5 km. The 2047-bin default sweep returned the 99.976 Hz
    bin instead, whose phase error grew linearly with range (measured 14.5
    deg at 2.5 km, 28.7 deg at 5 km)."""
    env = make_pekeris(name='ctl-pekeris', bathymetry=100.0)
    rcv = Receiver(depths=[50.0], ranges=np.linspace(2500.0, 5000.0, 6))
    src = Source(depths=25.0, frequencies=100.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        oasp = OASP(verbose=False).run(env, src, rcv)
    sco = Scooter(verbose=False).run(env, src, rcv)
    angles = np.angle(np.asarray(oasp.data)[0] / np.asarray(sco.data)[0],
                      deg=True)
    assert np.abs(angles).max() < 8.0, angles


@pytest.mark.requires_oases
def test_oast_returns_real_dB_tl(oast_field):
    """OAST's .plt carries only real TL, so its Field is real with
    ``unit='dB'`` and ``.dB`` is the data itself (no derivation)."""
    assert not oast_field.is_complex
    assert oast_field.kind == 'pressure'
    assert oast_field.unit == 'dB'
    np.testing.assert_array_equal(
        np.asarray(oast_field.dB), np.asarray(oast_field.data))


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
    TL is the modes binary's, RAM's flat-bathy dispatch lands on mpiramS."""
    assert kraken_field.backend == 'kraken'
    assert scooter_field.backend == 'scooter'
    assert ram_field.backend == 'mpirams'
    assert sparc_field.backend == 'sparc'


def test_ram_pe_reference_speed_is_the_settings_c0(ram_field):
    """RAM's PE expansion speed (m/s) is its run settings' ``c0``, stated
    once: neither ``pe_reference_speed`` nor ``c0`` is a metadata copy."""
    md = ram_field.metadata or {}
    assert 'pe_reference_speed' not in md, sorted(md)
    assert 'c0' not in md, sorted(md)
    c0 = float(ram_field.run_settings.engine.c0)
    assert 1400.0 < c0 < 1800.0   # a sound speed, not a flag or an index


# ── 3. shared geometry conventions ────────────────────────────────────────


def test_depth_swap_reciprocity_on_a_range_independent_channel():
    """kraken.md §7 "The same swap on a range-independent channel":
    swapping source and receiver depth on a
    range-INDEPENDENT channel reproduces the TL — the doc measured the swap
    at 0.000 dB, and pins the 3-12 dB non-reciprocity it documents to the
    range-dependent adiabatic sum, not to the solver. One engine suffices;
    Kraken is the engine the doc measured. 0.05 dB absorbs the mode
    tabulation grid changing with the receiver depth while sitting two
    orders below the range-dependent effect."""
    env = _pekeris()
    ranges = np.array([2000.0, 4000.0])
    forward = np.asarray(Kraken(verbose=False).run(
        env, Source(depths=30.0, frequencies=FREQ),
        Receiver(depths=np.array([80.0]), ranges=ranges),
        run_mode=RunMode.COHERENT_TL).dB)
    swapped = np.asarray(Kraken(verbose=False).run(
        env, Source(depths=80.0, frequencies=FREQ),
        Receiver(depths=np.array([30.0]), ranges=ranges),
        run_mode=RunMode.COHERENT_TL).dB)
    np.testing.assert_allclose(swapped, forward, rtol=0, atol=0.05)


@pytest.mark.slow
@pytest.mark.parametrize('model_cls', [Bellhop, Kraken, Scooter, RAM])
def test_degenerate_receiver_axes_stay_two_dimensional(model_cls):
    """source-receiver.md §2 "axis is kept, not dropped":
    shape follows the carrier for every field
    engine, and a degenerate axis is KEPT, not dropped — a vertical line
    array comes back ``(46, 1)`` and a horizontal one ``(1, 240)``, both
    still two-axis fields with ``{depth, range}`` coords."""
    env = _pekeris()
    src = _source()
    model = model_cls(verbose=False)
    vla = model.run(
        env, src, Receiver(depths=np.linspace(5.0, 95.0, 46), ranges=3000.0),
        run_mode=RunMode.COHERENT_TL)
    assert vla.data.shape == (46, 1)
    assert list(vla.coords) == ['depth', 'range']
    hla = model.run(
        env, src,
        Receiver(depths=60.0, ranges=np.linspace(200.0, 5000.0, 240)),
        run_mode=RunMode.COHERENT_TL)
    assert hla.data.shape == (1, 240)
    assert list(hla.coords) == ['depth', 'range']


class TestWaveEnginesAreReciprocal:
    """Swapping source and receiver depth must not change the field.

    Reciprocity is a property of the wave equation itself, so every
    wave-theoretic engine has to satisfy it in a range-independent guide.
    Measured on a 200 m Pekeris case at 150 Hz, 30 m <-> 150 m over 5 km:
    Kraken 0.004 dB, RAM 0.000 dB, Scooter 0.000 dB.

    Bellhop is deliberately excluded. ``beam_type='B'`` (the default) is NOT
    reciprocal — 0.9-1.0 dB on this case, and the gap does not shrink with
    beam count (0.91 dB at 2001 beams, 0.97 dB at 4001), so it is not a
    discretisation error. ``beam_type='G'`` (geometric hat) IS exactly
    reciprocal, and that is pinned below so the distinction cannot silently
    invert.

    The difference is in the beam SUM, not the ray amplitudes: JKPS Sect. 3.6.8
    proves the spreading function is symmetric under exchange of endpoints,
    and both letters are geometric beams (``bellhop.f90:309`` sends ``B`` to
    ``InfluenceGeoGaussianCart``, not to the Cerveny routines). ``B`` only
    swaps the hat shape function for a Gaussian of the same fan-derived width
    (JKPS Sect. 3.3.5.5). The hat vanishes at its neighbouring rays so the sum
    returns the ray-tube field that proof covers; the Gaussian overlaps its
    neighbours. See :class:`~uacpy.models.Bellhop`'s ``beam_type`` docs for the
    accuracy trade that makes ``B`` the default anyway.
    """

    Z1, Z2, RANGE, FREQ = 30.0, 150.0, 5000.0, 150.0

    @staticmethod
    def _env():
        import uacpy
        return uacpy.Environment(
            bathymetry=200.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=1.8, attenuation=0.5))

    def _tl(self, model, zs, zr):
        import warnings
        import numpy as np
        import uacpy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = model.run(self._env(),
                          uacpy.Source(depths=zs, frequencies=self.FREQ),
                          uacpy.Receiver(depths=[zr], ranges=[self.RANGE]))
        return float(np.asarray(f.to_dB().dB, dtype=float).ravel()[0])

    @pytest.mark.parametrize('engine', ['Kraken', 'RAM', 'Scooter'])
    def test_swapping_source_and_receiver_depth_returns_the_same_level(
            self, engine):
        from uacpy import models
        model = getattr(models, engine)(verbose=False)
        forward = self._tl(model, self.Z1, self.Z2)
        reverse = self._tl(model, self.Z2, self.Z1)
        assert abs(forward - reverse) < 0.05

    def test_the_geometric_hat_beam_is_reciprocal(self):
        from uacpy.models import Bellhop
        model = Bellhop(beam_type='G', n_beams=2001, verbose=False)
        assert abs(self._tl(model, self.Z1, self.Z2)
                   - self._tl(model, self.Z2, self.Z1)) < 0.05

    def test_the_default_gaussian_beam_is_not_reciprocal(self):
        # Pins the documented limitation, so a future change that makes B
        # reciprocal (or makes G stop being so) is noticed rather than assumed.
        from uacpy.models import Bellhop
        model = Bellhop(beam_type='B', n_beams=2001, verbose=False)
        assert abs(self._tl(model, self.Z1, self.Z2)
                   - self._tl(model, self.Z2, self.Z1)) > 0.3


@pytest.mark.requires_binary
def test_every_engine_reports_one_line_source_level():
    """Bellhop (4√π/√R raw), Kraken and Scooter (1/√(k0·R) raw) disagreed by
    4√(πk0) — 13 dB at 100 Hz — on ``source_type='line'``. All three now
    report unit amplitude at 1 m; the residual is ray-vs-wave physics.

    The levels are averaged over the cells finite in EVERY engine, not over
    each engine's own finite cells. This grid deliberately includes r = 0 and
    a sub-seafloor depth, where Kraken alone returns a value (contract 2
    above), so a per-engine ``nanmean`` would average three different cell
    sets and compare an in-water mean against one diluted by the much
    quieter transmitted field — a difference of grid, not of level
    convention. That artefact once masked a real 10·log10(k0) offset in
    Kraken's narrowband line-source branch: the as-written spread read
    1.8 dB while the two cells all three engines resolve differed by 4.5 dB.
    """
    env, point = _pekeris(), _source()
    src = Source(depths=point.depths, frequencies=point.frequencies,
                 source_type='line')
    rcv = _receiver()
    grids = {}
    for name, model in (('bellhop', Bellhop(verbose=False)),
                        ('kraken', Kraken(verbose=False)),
                        ('scooter', Scooter(verbose=False))):
        grids[name] = np.asarray(model.run(env, src, rcv).dB, dtype=float)
    common = np.all([np.isfinite(tl) for tl in grids.values()], axis=0)
    # The r > 0 in-water cells: enough of them that the mean is not one
    # sample, and the same cells for every engine.
    assert common.sum() >= 2, common
    levels = {name: float(np.mean(tl[common])) for name, tl in grids.items()}
    spread = max(levels.values()) - min(levels.values())
    assert spread < 2.0, levels


def test_every_quoted_model_helper_reference_resolves():
    """A ``base._name`` / ``_conventions._name`` pointer in a comment or
    docstring names a real function. The line-source normalisation helper
    moved from ``models/base.py`` to ``models/_conventions.py`` and four
    pointers kept naming ``base._line_source_unit_at_1m``, so a reader
    checking the line-source level (a units question) found nothing."""
    import importlib
    import pathlib
    import re as _re
    import uacpy
    modules = {name: importlib.import_module(f'uacpy.models.{name}')
               for name in ('base', '_conventions')}
    pattern = _re.compile(
        r'(?:``|\()(?:uacpy\.models\.)?(base|_conventions)\.(_\w+)(?:``|\))')
    root = pathlib.Path(uacpy.__file__).parent
    here = pathlib.Path(__file__).resolve()
    found, broken = 0, []
    for path in sorted(root.rglob('*.py')):
        # This file names the stale spelling in its own docstring.
        if 'third_party' in path.parts or path.resolve() == here:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for match in pattern.finditer(line):
                found += 1
                if not hasattr(modules[match.group(1)], match.group(2)):
                    broken.append(f"{path.relative_to(root)}:{number} "
                                  f"{match.group(0)}")
    assert found >= 4, f"the sweep found only {found} reference(s)"
    assert not broken, "references to no such helper:\n  " + \
        "\n  ".join(broken)
