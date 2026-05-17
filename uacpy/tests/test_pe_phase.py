"""Unit tests for ``models/_pe_phase.py``.

Each test computes the engineering travelling-wave conversion *both*
via the helper and via the pre-refactor inline math, then checks they
match to machine precision. Adding a new PE backend convention means
adding a new branch in ``_pe_phase.py`` and a corresponding test row
here — the helper is then guaranteed numerically identical to the
expected closed-form for every convention.
"""

import numpy as np
import pytest

from uacpy.models._pe_phase import (
    psi_to_travelling_wave,
    MPIRAMS, RAMS, RAMSURF,
)


def _rng(seed):
    g = np.random.default_rng(seed)
    return g


def test_mpirams_narrowband_matches_inline_math():
    """``_run_tl`` site: shape (n_z, n_r), no freq axis."""
    g = _rng(0xACED)
    psi = g.standard_normal((5, 4)) + 1j * g.standard_normal((5, 4))
    ranges = np.linspace(100.0, 1000.0, 4)

    out = psi_to_travelling_wave(
        psi, convention=MPIRAMS, ranges_m=ranges, range_axis=1,
    )

    # pre-refactor inline:
    #   p = conj(psif) · 4π · exp(-iπ/4) / √r
    scale = 4.0 * np.pi * np.exp(-1j * np.pi / 4.0) / np.sqrt(ranges)
    expected = np.conj(psi) * scale[np.newaxis, :]
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0)


def test_mpirams_broadband_matches_inline_math():
    """``_run_broadband`` site: psif shape (n_z, n_f, n_r); range on axis 2."""
    g = _rng(1)
    psif = g.standard_normal((3, 2, 4)) + 1j * g.standard_normal((3, 2, 4))
    ranges = np.linspace(100.0, 1000.0, 4)

    out = psi_to_travelling_wave(
        psif, convention=MPIRAMS, ranges_m=ranges, range_axis=2,
    )

    scale = 4.0 * np.pi * np.exp(-1j * np.pi / 4.0) / np.sqrt(ranges)
    expected = np.conj(psif) * scale[np.newaxis, np.newaxis, :]
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0)


def test_rams_broadband_no_carrier_no_radial():
    """``_run_collins_broadband`` site for ``rams0.5``: file already
    carries ψ·exp(+i k0 r); conj suffices. No radial scaling here —
    the Collins binaries write that in already."""
    g = _rng(2)
    H = g.standard_normal((4, 6, 8)) + 1j * g.standard_normal((4, 6, 8))
    out = psi_to_travelling_wave(
        H, convention=RAMS, ranges_m=np.linspace(100, 5000, 6),
        range_axis=1, freq_axis=2, apply_radial=False,
    )
    np.testing.assert_array_equal(out, np.conj(H))


def test_ramsurf_broadband_carrier_applied():
    """``_run_collins_broadband`` site for ``ramsurf1.5``: bare envelope
    needs explicit carrier ``exp(-i k0 r)`` multiplied in."""
    g = _rng(3)
    H = g.standard_normal((4, 6, 8)) + 1j * g.standard_normal((4, 6, 8))
    freqs = np.linspace(50.0, 200.0, 8)
    c0 = 1500.0
    k0 = 2.0 * np.pi * freqs / c0
    ranges = np.linspace(100.0, 5000.0, 6)

    out = psi_to_travelling_wave(
        H, convention=RAMSURF, ranges_m=ranges,
        range_axis=1, k0=k0, freq_axis=2, apply_radial=False,
    )

    carrier = np.exp(-1j * k0[None, None, :] * ranges[None, :, None])
    expected = np.conj(H) * carrier
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0)


def test_ramsurf_narrowband_scalar_k0():
    """Narrowband ramsurf path: scalar k0, no freq_axis."""
    g = _rng(4)
    psi = g.standard_normal((4, 6)) + 1j * g.standard_normal((4, 6))
    ranges = np.linspace(100.0, 5000.0, 6)
    k0_scalar = 2.0 * np.pi * 100.0 / 1500.0

    out = psi_to_travelling_wave(
        psi, convention=RAMSURF, ranges_m=ranges,
        range_axis=1, k0=k0_scalar, apply_radial=False,
    )

    carrier = np.exp(-1j * k0_scalar * ranges)
    expected = np.conj(psi) * carrier[np.newaxis, :]
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0)


def test_unknown_convention_raises():
    psi = np.zeros((2, 2), dtype=complex)
    with pytest.raises(ValueError, match="unknown PE convention"):
        psi_to_travelling_wave(
            psi, convention='lytaev_v2', ranges_m=np.array([1.0, 2.0]),
            range_axis=1,
        )


def test_ramsurf_without_k0_raises():
    psi = np.zeros((2, 2), dtype=complex)
    with pytest.raises(ValueError, match="requires k0"):
        psi_to_travelling_wave(
            psi, convention=RAMSURF, ranges_m=np.array([1.0, 2.0]),
            range_axis=1,
        )


def test_apply_radial_true_divides_by_sqrt_r():
    """Default behaviour: 1/√r is applied. Compare against apply_radial=False
    × manual 1/√r."""
    g = _rng(5)
    psi = g.standard_normal((3, 4)) + 1j * g.standard_normal((3, 4))
    ranges = np.linspace(100.0, 1000.0, 4)

    with_radial = psi_to_travelling_wave(
        psi, convention=MPIRAMS, ranges_m=ranges, range_axis=1,
        apply_radial=True,
    )
    without_radial = psi_to_travelling_wave(
        psi, convention=MPIRAMS, ranges_m=ranges, range_axis=1,
        apply_radial=False,
    )

    np.testing.assert_allclose(
        with_radial,
        without_radial / np.sqrt(ranges)[np.newaxis, :],
        atol=1e-12, rtol=0,
    )
