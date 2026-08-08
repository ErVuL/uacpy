"""Independent, hand-written tests for the matched-field processing layer.

The decisive correctness oracle is physical: the Python modal-sum replica
(:func:`synthesize_replica`) must reproduce ``field.exe``'s complex pressure
(``Kraken.run`` COHERENT_TL) up to a single global complex scalar. The
processor tests inject a known source and check the ambiguity surface peaks
at the true location.
"""

import numpy as np
import pytest

import uacpy
from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError
from uacpy import Bottom
from uacpy.models import Kraken

from uacpy.sonar.matched_field import (
    synthesize_replica,
    replica_bank,
    csdm,
    bartlett,
    mvdr,
)

# The fixtures run kraken.exe / krakenc.exe + field.exe.
pytestmark = pytest.mark.requires_binary


@pytest.fixture(scope="module")
def pekeris():
    """A simple isovelocity Pekeris waveguide: modes + the field.exe field."""
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
    src = uacpy.Source(depths=25.0, frequencies=150.0)
    rcv = uacpy.Receiver(
        depths=np.linspace(1, 99, 99), ranges=np.linspace(200, 5000, 80)
    )
    kraken = Kraken(verbose=False)
    modes = kraken.compute_modes(env, src)
    field = kraken.run(env, src, rcv)  # COHERENT_TL -> complex pressure
    return dict(modes=modes, field=field, src=src, rcv=rcv)


# ── replica synthesis: the physical oracle ──────────────────────────────

def test_replica_reproduces_field_exe_up_to_global_scalar(pekeris):
    modes, field, rcv = pekeris["modes"], pekeris["field"], pekeris["rcv"]
    syn = synthesize_replica(
        modes, src_depth=25.0, ranges=rcv.ranges, array_depths=rcv.depths
    )
    P = field.data  # (n_depth, n_range)
    assert syn.shape == P.shape
    corr = abs(np.vdot(syn, P)) / (np.linalg.norm(syn) * np.linalg.norm(P))
    assert corr > 0.999, f"replica vs field.exe normalized corr = {corr}"


def test_replica_vector_has_one_entry_per_array_element(pekeris):
    modes = pekeris["modes"]
    z_arr = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
    e = synthesize_replica(modes, src_depth=40.0, ranges=2000.0, array_depths=z_arr)
    assert e.shape == (z_arr.size,)
    assert np.iscomplexobj(e)


def test_replica_obeys_source_receiver_reciprocity(pekeris):
    # The modal sum is symmetric in source/receiver depth: the pressure at z_b
    # from a source at z_a equals the pressure at z_a from a source at z_b.
    modes = pekeris["modes"]
    p_ab = synthesize_replica(modes, src_depth=20.0, ranges=2500.0, array_depths=70.0)
    p_ba = synthesize_replica(modes, src_depth=70.0, ranges=2500.0, array_depths=20.0)
    assert p_ab == pytest.approx(p_ba, rel=1e-12)


def test_synthesize_replica_rejects_nonpositive_range(pekeris):
    modes = pekeris["modes"]
    with pytest.raises(ConfigurationError):
        synthesize_replica(modes, src_depth=25.0, ranges=0.0, array_depths=50.0)


def test_replica_reproduces_field_exe_with_complex_modes():
    # krakenc -> complex k and complex phi: exercises the complex-interp path.
    env = uacpy.Environment(
        name="Pekeris-attenuating",
        bathymetry=100.0,
        ssp=SoundSpeedProfile.from_pairs(np.array([[0, 1500.0], [100, 1500.0]])),
        bottom=Bottom.from_halfspaces(
            np.array([0.0]),
            sound_speed=np.array([1700.0]),
            density=np.array([1.6]),
            attenuation=np.array([0.5]),
            acoustic_type="half-space",
        ),
    )
    src = uacpy.Source(depths=30.0, frequencies=120.0)
    rcv = uacpy.Receiver(depths=np.linspace(1, 99, 90), ranges=np.linspace(300, 4000, 60))
    kraken = Kraken(backend="krakenc", verbose=False)
    modes = kraken.compute_modes(env, src)
    field = kraken.run(env, src, rcv)
    assert np.iscomplexobj(modes.phi)
    syn = synthesize_replica(modes, 30.0, rcv.ranges, rcv.depths)
    P = field.data
    corr = abs(np.vdot(syn, P)) / (np.linalg.norm(syn) * np.linalg.norm(P))
    assert corr > 0.999, f"krakenc replica vs field.exe corr = {corr}"


# ── CSDM ────────────────────────────────────────────────────────────────

def test_csdm_is_hermitian_and_square():
    rng = np.random.default_rng(0)
    snaps = rng.standard_normal((6, 20)) + 1j * rng.standard_normal((6, 20))
    K = csdm(snaps)
    assert K.shape == (6, 6)
    assert np.allclose(K, K.conj().T)


# ── Bartlett processor ──────────────────────────────────────────────────

def test_bartlett_peaks_at_true_source(pekeris):
    modes = pekeris["modes"]
    z_arr = np.linspace(5, 95, 16)            # 16-element vertical array
    cand_z = np.linspace(5, 95, 46)
    cand_r = np.linspace(500, 5000, 60)
    bank = replica_bank(modes, z_arr, cand_z, cand_r)   # (N, nz, nr)

    iz, ir = 18, 25                            # true source grid cell
    d = bank[:, iz, ir]                        # noise-free data = that replica
    K = csdm(d[:, None])
    surf = bartlett(K, bank)
    assert surf.shape == (cand_z.size, cand_r.size)
    peak = np.unravel_index(np.argmax(surf), surf.shape)
    assert peak == (iz, ir)


def test_bartlett_is_normalized_to_unit_at_perfect_match(pekeris):
    modes = pekeris["modes"]
    z_arr = np.linspace(5, 95, 16)
    cand_z = np.linspace(5, 95, 20)
    cand_r = np.linspace(500, 5000, 20)
    bank = replica_bank(modes, z_arr, cand_z, cand_r)
    d = bank[:, 7, 11]
    surf = bartlett(csdm(d[:, None]), bank)
    assert surf.min() >= -1e-9 and surf.max() <= 1.0 + 1e-9
    assert surf[7, 11] == pytest.approx(1.0, abs=1e-6)


# ── MVDR processor ──────────────────────────────────────────────────────

def test_mvdr_peaks_at_true_source(pekeris):
    modes = pekeris["modes"]
    z_arr = np.linspace(5, 95, 16)
    cand_z = np.linspace(5, 95, 40)
    cand_r = np.linspace(500, 5000, 50)
    bank = replica_bank(modes, z_arr, cand_z, cand_r)

    iz, ir = 22, 30
    d = bank[:, iz, ir]
    K = csdm(d[:, None])
    surf = mvdr(K, bank, loading=1e-2)         # loading needed for rank-1 K
    peak = np.unravel_index(np.argmax(surf), surf.shape)
    assert peak == (iz, ir)


@pytest.mark.parametrize("processor", ["bartlett", "mvdr"])
def test_localizes_offgrid_source_from_noisy_snapshots(pekeris, processor):
    # Truth sits *between* grid nodes; recover it to the nearest cell at 10 dB
    # SNR from 50 snapshots with random source phase + sensor noise.
    modes = pekeris["modes"]
    z_arr = np.linspace(5, 95, 16)
    true_z, true_r = 62.0, 3200.0
    e_true = synthesize_replica(modes, true_z, true_r, z_arr)

    rng = np.random.default_rng(1)
    L = 50
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, L))
    sig = e_true[:, None] * phases
    npow = np.mean(np.abs(e_true) ** 2) / 10 ** (10.0 / 10)
    noise = np.sqrt(npow / 2) * (
        rng.standard_normal((16, L)) + 1j * rng.standard_normal((16, L))
    )
    K = csdm(sig + noise)

    cand_z = np.linspace(5, 95, 91)
    cand_r = np.linspace(500, 5000, 91)
    bank = replica_bank(modes, z_arr, cand_z, cand_r)
    surf = bartlett(K, bank) if processor == "bartlett" else mvdr(K, bank, loading=1e-2)

    iz, ir = np.unravel_index(np.argmax(surf), surf.shape)
    # nearest grid node to truth
    assert abs(cand_z[iz] - true_z) <= (cand_z[1] - cand_z[0])
    assert abs(cand_r[ir] - true_r) <= (cand_r[1] - cand_r[0])


def test_replica_decays_under_either_imag_k_sign():
    """Raw Kraken carries Im(k) < 0; ``Modes.with_attenuation`` builds
    Im(k) > 0. A passive medium can only attenuate, so the replica must decay
    with range under BOTH. Regression for the x1.75e11 grow-with-range bug that
    put an MFP peak at 14 m / 5.75 km instead of 42 m / 10 km."""
    from uacpy.core.results import Modes
    from uacpy.sonar.matched_field import synthesize_replica
    depths = np.linspace(0.0, 100.0, 51)
    phi = np.column_stack([np.sin((m + 0.5) * np.pi * depths / 100.0)
                           for m in range(3)])
    k_real = np.array([0.0628, 0.0625, 0.0619])
    r = np.array([1000.0, 20000.0])
    for sign in (+1.0, -1.0):
        modes = Modes(k=k_real + sign * 1j * 3e-4, phi=phi, depths=depths,
                      model='Test', frequencies=15.0)
        p = synthesize_replica(modes, 25.0, r, np.array([50.0]))
        env = np.abs(np.asarray(p).ravel()) * np.sqrt(r)   # drop 1/sqrt(r)
        assert env[-1] < env[0], (
            f"replica grew with range for Im(k) sign {sign:+.0f}")


def test_replica_carries_the_per_mode_hankel_weight():
    """The far-field modal sum weights each mode by ``1/sqrt(k_m)`` (the
    asymptotic Hankel form). Dropping it leaves a replica that still decays and
    still normalises, so only a mode-resolved check sees it: with two modes of
    equal shape amplitude the ratio of their contributions is exactly
    ``sqrt(k2/k1)``."""
    from uacpy.core.results import Modes
    from uacpy.sonar.matched_field import synthesize_replica
    depths = np.linspace(0.0, 100.0, 51)
    k = np.array([0.0628, 0.0400])
    r = np.array([5000.0])
    z_r = np.array([50.0])
    # One mode at a time, identical shapes -> the only difference is 1/sqrt(k).
    amps = []
    for m in range(2):
        phi = np.sin(np.pi * depths / 100.0)[:, None]
        modes = Modes(k=np.array([k[m]]), phi=phi, depths=depths,
                      model='Test', frequencies=15.0)
        p = synthesize_replica(modes, 25.0, r, z_r)
        amps.append(np.abs(np.asarray(p).ravel()[0]))
    assert amps[0] / amps[1] == pytest.approx(np.sqrt(k[1] / k[0]), rel=1e-9)
