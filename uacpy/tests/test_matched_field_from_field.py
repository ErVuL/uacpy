"""Independent, hand-written tests for the model-agnostic MFP replica adapter
:func:`replica_bank_from_field`.

The contract tests are binary-free: they build :class:`Field` / :class:`ResultStack`
objects from known complex arrays and check the adapter reshapes them into the
exact ``(N, *candidate_grid)`` bank that :func:`bartlett` / :func:`mvdr` consume.
A single ``requires_binary`` test closes the physical loop: a KRAKEN ``field.exe``
pressure run, routed through the adapter, must localize a planted source — and
must agree with the modal-sum bank from :func:`replica_bank`.
"""

import numpy as np
import pytest

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results.field import Field, ResultStack
from uacpy.core.results.array_products import Covariance, Replicas
from uacpy.sonar.matched_field import (
    replica_bank,
    replica_bank_from_field,
    csdm,
    bartlett,
    mvdr,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _pressure_field(data, *, depth, range):
    """Build a coherent {depth, range} pressure Field from a (N, R) array."""
    return Field(data=np.asarray(data), coords={"depth": depth, "range": range})


def _random_bank(n_arr, n_cz, n_cr, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((n_arr, n_cz, n_cr))
        + 1j * rng.standard_normal((n_arr, n_cz, n_cr))
    )


# ── ResultStack path ─────────────────────────────────────────────────────

def test_resultstack_reproduces_bank_exactly():
    z_arr = np.linspace(5, 95, 8)
    cand_z = np.linspace(10, 90, 6)
    cand_r = np.linspace(500, 4000, 12)
    P = _random_bank(z_arr.size, cand_z.size, cand_r.size)

    slabs = [
        _pressure_field(P[:, i, :], depth=z_arr, range=cand_r)
        for i in range(cand_z.size)
    ]
    stack = ResultStack(slabs, cand_z, coordinate_name="source_depth")

    bank = replica_bank_from_field(stack, array_depths=z_arr)
    assert bank.shape == (z_arr.size, cand_z.size, cand_r.size)
    assert np.array_equal(bank, P)


def test_resultstack_rejects_mismatched_range_axis():
    # Same length, different range vectors across slabs → silent-wrong bank.
    z_arr = np.linspace(5, 95, 4)
    cand_z = np.array([20.0, 40.0])
    slabs = [
        _pressure_field(_random_bank(4, 1, 6, seed=0)[:, 0, :],
                        depth=z_arr, range=np.linspace(500, 4000, 6)),
        _pressure_field(_random_bank(4, 1, 6, seed=1)[:, 0, :],
                        depth=z_arr, range=np.linspace(400, 3000, 6)),  # differs
    ]
    stack = ResultStack(slabs, cand_z, coordinate_name="source_depth")
    with pytest.raises(ConfigurationError, match="range"):
        replica_bank_from_field(stack)


def test_resultstack_rejects_mismatched_depth_axis():
    # Different array geometry across slabs, array_depths not supplied.
    cand_z = np.array([20.0, 40.0])
    cand_r = np.linspace(500, 4000, 6)
    slabs = [
        _pressure_field(_random_bank(4, 1, 6, seed=0)[:, 0, :],
                        depth=np.linspace(5, 95, 4), range=cand_r),
        _pressure_field(_random_bank(4, 1, 6, seed=1)[:, 0, :],
                        depth=np.linspace(10, 90, 4), range=cand_r),  # differs
    ]
    stack = ResultStack(slabs, cand_z, coordinate_name="source_depth")
    with pytest.raises(ConfigurationError, match="depth"):
        replica_bank_from_field(stack)


def test_single_slab_stack_is_one_candidate_depth():
    z_arr = np.linspace(5, 95, 5)
    cand_r = np.linspace(500, 4000, 7)
    data = _random_bank(5, 1, 7)[:, 0, :]
    stack = ResultStack([_pressure_field(data, depth=z_arr, range=cand_r)],
                        np.array([30.0]), coordinate_name="source_depth")
    bank = replica_bank_from_field(stack, array_depths=z_arr)
    assert bank.shape == (5, 1, 7)
    assert np.array_equal(bank[:, 0, :], data)


def test_bank_is_writeable_and_independent_of_field():
    # field.p is a read-only view; the returned bank must be an owned, writeable
    # copy — mutating it must not corrupt the source Field.
    z_arr = np.linspace(5, 95, 4)
    cand_r = np.linspace(500, 4000, 5)
    data = _random_bank(4, 1, 5)[:, 0, :]
    field = _pressure_field(data, depth=z_arr, range=cand_r)
    bank = replica_bank_from_field(field, array_depths=z_arr)
    assert bank.flags.writeable
    bank[0, 0] = 12345.0 + 0j
    assert np.array_equal(field.p, data)          # source untouched


def test_noncanonical_axis_order_depth_last():
    # A {source_depth, range, depth} field (depth last) must still produce the
    # canonical (N, n_cand_depth, n_cand_range) bank.
    z_arr = np.linspace(5, 95, 6)
    cand_z = np.linspace(10, 90, 4)
    cand_r = np.linspace(500, 4000, 7)
    # data laid out as (source_depth, range, depth)
    data = _random_bank(cand_z.size, cand_r.size, z_arr.size)  # (sd, r, N)
    field = Field(data=data,
                  coords={"source_depth": cand_z, "range": cand_r, "depth": z_arr})
    bank = replica_bank_from_field(field, array_depths=z_arr)
    assert bank.shape == (z_arr.size, cand_z.size, cand_r.size)
    # bank[n, i, j] must equal data[i, j, n]
    assert np.array_equal(bank, np.moveaxis(data, 2, 0))


def test_resultstack_rejects_non_source_depth_axis():
    z_arr = np.linspace(5, 95, 4)
    cand_r = np.linspace(500, 4000, 5)
    slabs = [
        _pressure_field(_random_bank(4, 1, 5, seed=k)[:, 0, :], depth=z_arr, range=cand_r)
        for k in range(3)
    ]
    stack = ResultStack(slabs, np.array([100.0, 200.0, 300.0]),
                        coordinate_name="frequency")
    with pytest.raises(ConfigurationError, match="source_depth"):
        replica_bank_from_field(stack)


# ── single-Field paths ───────────────────────────────────────────────────

def test_single_multisource_field_moves_depth_to_front():
    z_arr = np.linspace(5, 95, 7)
    cand_z = np.linspace(10, 90, 5)
    cand_r = np.linspace(500, 4000, 9)
    # canonical {source_depth, depth, range} layout: data (n_cz, N, n_cr)
    data = _random_bank(cand_z.size, z_arr.size, cand_r.size)  # (cz, N, cr)
    field = Field(
        data=data,
        coords={"source_depth": cand_z, "depth": z_arr, "range": cand_r},
    )
    bank = replica_bank_from_field(field, array_depths=z_arr)
    # depth axis (was position 1) → position 0
    assert bank.shape == (z_arr.size, cand_z.size, cand_r.size)
    assert np.array_equal(bank, np.moveaxis(data, 1, 0))


def test_single_depth_range_field_is_range_only_bank():
    z_arr = np.linspace(5, 95, 6)
    cand_r = np.linspace(500, 4000, 10)
    data = _random_bank(z_arr.size, 1, cand_r.size)[:, 0, :]  # (N, R)
    field = _pressure_field(data, depth=z_arr, range=cand_r)
    bank = replica_bank_from_field(field, array_depths=z_arr)
    assert bank.shape == (z_arr.size, cand_r.size)
    assert np.array_equal(bank, data)


# ── feeds the processors and localizes ───────────────────────────────────

@pytest.mark.parametrize("processor", ["bartlett", "mvdr"])
def test_adapter_bank_localizes_planted_source(processor):
    z_arr = np.linspace(5, 95, 12)
    cand_z = np.linspace(10, 90, 14)
    cand_r = np.linspace(500, 4000, 20)
    P = _random_bank(z_arr.size, cand_z.size, cand_r.size, seed=3)

    slabs = [
        _pressure_field(P[:, i, :], depth=z_arr, range=cand_r)
        for i in range(cand_z.size)
    ]
    stack = ResultStack(slabs, cand_z, coordinate_name="source_depth")
    bank = replica_bank_from_field(stack, array_depths=z_arr)

    iz, ir = 9, 13
    d = bank[:, iz, ir]                       # noise-free data = that replica
    K = csdm(d[:, None])
    surf = bartlett(K, bank) if processor == "bartlett" else mvdr(K, bank, loading=1e-2)
    assert surf.shape == (cand_z.size, cand_r.size)
    assert np.unravel_index(np.argmax(surf), surf.shape) == (iz, ir)


# ── error contracts ──────────────────────────────────────────────────────

def test_rejects_real_tl_field():
    z_arr = np.linspace(5, 95, 5)
    cand_r = np.linspace(500, 4000, 7)
    tl = Field(data=np.abs(_random_bank(5, 1, 7)[:, 0, :]),  # real → kind 'tl'
               coords={"depth": z_arr, "range": cand_r})
    with pytest.raises(ConfigurationError, match="kind='pressure'|pressure"):
        replica_bank_from_field(tl)


def test_rejects_frequency_axis():
    z_arr = np.linspace(5, 95, 4)
    cand_r = np.linspace(500, 4000, 6)
    freqs = np.array([100.0, 150.0, 200.0])
    data = (np.random.default_rng(0).standard_normal((4, 6, 3))
            + 1j * np.random.default_rng(1).standard_normal((4, 6, 3)))
    tf = Field(data=data, coords={"depth": z_arr, "range": cand_r, "frequency": freqs})
    with pytest.raises(ConfigurationError):
        replica_bank_from_field(tf)


def test_rejects_array_depths_mismatch():
    z_arr = np.linspace(5, 95, 6)
    cand_r = np.linspace(500, 4000, 8)
    field = _pressure_field(_random_bank(6, 1, 8)[:, 0, :], depth=z_arr, range=cand_r)
    with pytest.raises(ConfigurationError, match="array_depths"):
        replica_bank_from_field(field, array_depths=np.linspace(5, 95, 7))


def test_rejects_field_without_depth_axis():
    cand_z = np.linspace(10, 90, 5)
    cand_r = np.linspace(500, 4000, 8)
    data = _random_bank(cand_z.size, 1, cand_r.size)[:, 0, :]
    field = Field(data=data, coords={"source_depth": cand_z, "range": cand_r})
    with pytest.raises(ConfigurationError, match="depth"):
        replica_bank_from_field(field)


# ── cross-check vs the in-tree OASN processor (toolbox-grade reference) ──
#
# OASN (OASES) is uacpy's established, toolbox-grade MFP: it computes the
# replica Green's functions itself (`Replicas`) and processes them with its own
# Bartlett/MVDR (`Covariance.bartlett/mvdr`). The strongest validation of the
# model-agnostic adapter is therefore a *processor-equivalence* check: feed
# OASN's OWN replica vectors through `replica_bank_from_field` + our processors
# and require the SAME ambiguity surface. This isolates our code from any
# cross-model physics mismatch — given identical Green's functions, the two MFP
# pipelines must agree. Conventions differ by exactly one scalar: OASN Bartlett
# is wᴴCw, ours divides by tr(K); OASN MVDR is unnormalised, ours scales max→1.

def _replicas_to_stack(rep, *, freq_idx=0):
    """Repackage an OASN ``Replicas`` (n_f, nz, nx, ny=1, N) as a ResultStack of
    ``{depth, range}`` pressure Fields over ``source_depth`` — the adapter input.
    Depth-range MFP only (requires the horizontal ``y`` axis to be a singleton).
    """
    R = np.asarray(rep.replicas)
    if R.shape[3] != 1:
        raise ConfigurationError("cross-check is depth-range only (n_yr == 1)")
    z_arr = (rep.receiver_positions[:, 2] if rep.receiver_positions is not None
             else np.arange(R.shape[4], dtype=float))
    cand_r = rep.replica_x
    slabs = [
        Field(data=R[freq_idx, iz, :, 0, :].T,            # (N, nx)
              coords={"depth": z_arr, "range": cand_r})
        for iz in range(R.shape[1])
    ]
    return ResultStack(slabs, rep.replica_z, coordinate_name="source_depth")


def _assert_adapter_matches_oasn_processor(rep, *, rtol=1e-9, atol=1e-12):
    """Adapter + our bartlett/mvdr reproduce OASN's processor on identical
    replicas: same bank, same peak, same surface up to the known scalar.

    ``rtol``/``atol`` bound the surface-value match. The decisive checks (exact
    bank reproduction, identical argmax) are tolerance-independent; the value
    comparison is loosened for real OASN output, whose ``.rpo`` replicas carry
    single-precision (~1e-7) Fortran round-off.
    """
    R = np.asarray(rep.replicas)
    N = R.shape[4]

    # 1. reshaping fidelity: our bank IS the OASN replica tensor, reordered.
    our_bank = replica_bank_from_field(_replicas_to_stack(rep))   # (N, nz, nx)
    oasn_bank = np.moveaxis(R[0, :, :, 0, :], -1, 0)
    assert our_bank.shape == (N, R.shape[1], R.shape[2])
    assert np.array_equal(our_bank, oasn_bank)

    # 2. plant a rank-1 + white-noise CSDM from one OASN replica column.
    iz0, ix0 = R.shape[1] // 3, R.shape[2] // 2
    w = oasn_bank[:, iz0, ix0]
    sig = np.outer(w, w.conj())
    K = sig + 0.05 * (np.trace(sig).real / N) * np.eye(N)
    cov = Covariance(covariance=K[None])

    # 3. Bartlett: peaks coincide; ref == ours · tr(K).
    ref_b = cov.bartlett(rep)[0, :, :, 0]
    our_b = bartlett(K, our_bank)
    assert np.unravel_index(np.argmax(our_b), our_b.shape) == (iz0, ix0)
    assert np.unravel_index(np.argmax(ref_b), ref_b.shape) == (iz0, ix0)
    np.testing.assert_allclose(our_b * np.trace(K).real, ref_b, rtol=rtol, atol=atol)

    # 4. MVDR (same loading): peaks coincide; ours == ref normalised to max 1.
    L = 1e-2
    ref_m = cov.mvdr(rep, diagonal_loading=L)[0, :, :, 0]
    our_m = mvdr(K, our_bank, loading=L)
    assert np.unravel_index(np.argmax(our_m), our_m.shape) == (iz0, ix0)
    np.testing.assert_allclose(
        our_m, ref_m / np.nanmax(ref_m), rtol=max(rtol, 1e-6), atol=max(atol, 1e-9))


def test_adapter_equivalent_to_oasn_processor_synthetic():
    # Binary-free: a synthetic OASN-style replica tensor drives the full
    # cross-check without needing the OASES binaries.
    rng = np.random.default_rng(7)
    N, nz, nx = 10, 12, 18
    R = (rng.standard_normal((1, nz, nx, 1, N))
         + 1j * rng.standard_normal((1, nz, nx, 1, N)))
    rcv_pos = np.zeros((N, 3))
    rcv_pos[:, 2] = np.linspace(5, 95, N)
    rep = Replicas(
        replicas=R,
        replica_z=np.linspace(10, 90, nz),
        replica_x=np.linspace(200, 4000, nx),
        replica_y=np.array([0.0]),
        receiver_positions=rcv_pos,
    )
    _assert_adapter_matches_oasn_processor(rep)


@pytest.mark.requires_oases
def test_adapter_equivalent_to_oasn_processor_real():
    # Same cross-check on a real OASN replica field (requires OASES binaries).
    import uacpy
    from uacpy.core.environment import SoundSpeedProfile
    from uacpy import Bottom
    from uacpy.models import OASN

    env = uacpy.Environment(
        name="Pekeris",
        bathymetry=100.0,
        ssp=SoundSpeedProfile.from_pairs(np.array([[0, 1500.0], [100, 1500.0]])),
        bottom=Bottom.from_halfspaces(
            np.array([0.0]),
            sound_speed=np.array([1800.0]),
            density=np.array([1.8]),
            attenuation=np.array([0.2]),
            acoustic_type="half-space",
        ),
    )
    src = uacpy.Source(depths=20.0, frequencies=150.0)
    rcv = uacpy.Receiver(depths=np.linspace(10, 90, 16), ranges=0.0)
    oasn = OASN(
        zmin=10.0, zmax=90.0, nz=12,
        xmin=0.2, xmax=4.0, nx=18, ny=1,   # km on the OASN constructor
        verbose=False,
    )
    rep = oasn.compute_replicas(env, src, rcv)
    assert rep.replicas.shape[3] == 1            # depth-range grid
    # OASN .rpo replicas are single-precision Fortran output: loosen the
    # surface-value tolerance accordingly (peak/bank checks stay exact).
    _assert_adapter_matches_oasn_processor(rep, rtol=1e-5, atol=1e-9)


# ── physical loop: field.exe pressure → adapter (requires binaries) ──────

@pytest.mark.requires_binary
def test_adapter_matches_modal_bank_and_localizes():
    import uacpy
    from uacpy.core.environment import SoundSpeedProfile
    from uacpy import Bottom
    from uacpy.models import Kraken

    env = uacpy.Environment(
        name="Pekeris",
        bathymetry=100.0,
        ssp=SoundSpeedProfile.from_pairs(np.array([[0, 1500.0], [100, 1500.0]])),
        bottom=Bottom.from_halfspaces(
            np.array([0.0]),
            sound_speed=np.array([1800.0]),
            density=np.array([1.8]),
            attenuation=np.array([0.2]),
            acoustic_type="half-space",
        ),
    )
    z_arr = np.linspace(5, 95, 16)
    cand_z = np.linspace(10, 90, 8)
    cand_r = np.linspace(500, 5000, 40)

    kraken = Kraken(verbose=False)
    modes = kraken.compute_modes(env, uacpy.Source(depths=cand_z[0], frequencies=150.0))

    # Build the bank the generic way: one field.exe run per candidate depth,
    # receivers AT the array elements over the candidate ranges.
    slabs = []
    for zs in cand_z:
        src = uacpy.Source(depths=float(zs), frequencies=150.0)
        rcv = uacpy.Receiver(depths=z_arr, ranges=cand_r)
        slabs.append(kraken.run(env, src, rcv))     # COHERENT_TL → {depth, range} pressure
    stack = ResultStack(slabs, cand_z, coordinate_name="source_depth")

    bank_field = replica_bank_from_field(stack, array_depths=z_arr)
    bank_modal = replica_bank(modes, z_arr, cand_z, cand_r)
    assert bank_field.shape == bank_modal.shape

    # Per-replica agreement up to the single global complex scalar both share:
    # unit-normalised columns must be collinear (|<a,b>| ≈ 1).
    Ef = bank_field.reshape(z_arr.size, -1)
    Em = bank_modal.reshape(z_arr.size, -1)
    Ef /= np.linalg.norm(Ef, axis=0, keepdims=True)
    Em /= np.linalg.norm(Em, axis=0, keepdims=True)
    coll = np.abs(np.sum(Ef.conj() * Em, axis=0))
    assert coll.min() > 0.999, f"min column collinearity = {coll.min()}"

    # And the field.exe bank localizes a planted source.
    iz, ir = 5, 22
    d = bank_field[:, iz, ir]
    surf = bartlett(csdm(d[:, None]), bank_field)
    assert np.unravel_index(np.argmax(surf), surf.shape) == (iz, ir)
