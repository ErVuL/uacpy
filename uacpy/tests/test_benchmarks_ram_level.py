"""Tier-1 numerical benchmarks for the RAM family's absolute LEVEL against
the analytic Pekeris modal sum, per backend.

``test_benchmarks_analytic.test_ram_tl_matches_pekeris_modal_sum`` bounds
RAM at a 1.5 dB median and ``test_ram_backends.TestRamPekerisReference`` at
1.5–2.0 dB window medians against Kraken (measured 2026-09-19: the fields
sit at 0.5 dB). This file pins two things those bounds cannot:

* the SHAPE of the field per backend — the deviation of ``RAM − modal sum``
  from its own median, which is what a wrapper bug in the grid, the
  reference speed or the starter shows as;
* the ABSOLUTE level at a case where the PE error itself is small, found
  by sweeping (frequency, depth): 100 Hz in 100 m (seven trapped modes).
  Higher frequency in the same water is *worse* for this Padé PE at the
  1500/1800 m/s contrast (200 Hz: 3.5 dB median on the auto grid, 1.9 dB
  on a dr=2 m / dz=0.25 m grid), and 100 Hz in 200 m worse again (1.25 dB) —
  the mode count, not the resolution, drives the error.

The reference is ``pekeris_modal_tl`` from ``test_benchmarks_analytic``
(Porter, *KRAKEN Normal Mode Program*, eq. 2.19), which Kraken reproduces to
0.02 dB; using it rather than a Kraken field keeps the reference independent
of every engine. Every bound is a multiple of a measured value recorded in
the fixer ledger.
"""
import warnings

import numpy as np
import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.requires_binary]

from uacpy import (Environment, SoundSpeedProfile, BoundaryProperties, Source,
                   Receiver, RunMode, RAM)
from uacpy.tests.test_benchmarks_analytic import (
    pekeris_modal_tl, RHO_WATER, C_W, C_B, RHO_B)

# Every 500 m from 2 to 10 km at three depths: 51 cells, enough that a
# single interference null (the one at z = 50 m, r = 2 km reads 4.8-5.6 dB
# off on every fluid backend) cannot move a median.
_RANGES = np.arange(2000.0, 10001.0, 500.0)
# (frequency, water depth, source depth, receiver depths)
_STANDARD = (50.0, 100.0, 25.0, [30.0, 50.0, 75.0])   # the analytic suite's case
_BEST = (100.0, 100.0, 25.0, [30.0, 50.0, 75.0])      # the well-resolved case
# rams needs an elastic seabed. 50 m/s of shear is a near-fluid bottom whose
# Rayleigh coefficient differs from the fluid one by a negligible amount at
# every trapped-mode angle — measured: rams on it matches the FLUID modal
# sum to 0.04 dB median — and its λ_s/8 grid cap (dz = 0.071 m) is what
# makes the run 5 s rather than 0.2 s.
_RAMS_SHEAR = 50.0


def _pekeris(depth, altimetry=None, shear=None):
    if shear is None:
        bottom = BoundaryProperties(acoustic_type='half-space', sound_speed=C_B,
                                    density=RHO_B, attenuation=0.0)
    else:
        bottom = BoundaryProperties(acoustic_type='half-space', sound_speed=C_B,
                                    density=RHO_B, attenuation=0.0,
                                    shear_speed=shear, shear_attenuation=0.0)
    return Environment(
        water_density=RHO_WATER, bathymetry=depth,
        ssp=SoundSpeedProfile.from_pairs([(0.0, C_W), (depth, C_W)]),
        bottom=bottom, altimetry=altimetry)


def _ram_minus_modal_sum(backend, case):
    """``RAM(backend) − pekeris_modal_tl`` in dB over the depth x range
    table of ``case``, flattened; asserts the dispatcher used ``backend``."""
    f, depth, z_s, z_r = case
    altimetry = ([(0.0, 0.0), (_RANGES.max() * 1.05, 0.0)]
                 if backend == 'ramsurf' else None)
    shear = _RAMS_SHEAR if backend == 'rams' else None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        field = RAM(verbose=False, backend=backend, timeout=600).run(
            _pekeris(depth, altimetry, shear),
            Source(depths=z_s, frequencies=f),
            Receiver(depths=z_r, ranges=_RANGES), run_mode=RunMode.COHERENT_TL)
    assert field.backend == backend, field.backend
    tl = np.asarray(field.dB).reshape(len(z_r), _RANGES.size)
    ana = np.array([pekeris_modal_tl(z_s, zr, _RANGES, f, depth,
                                     C_W, C_B, RHO_WATER, RHO_B) for zr in z_r])
    d = (tl - ana).ravel()
    assert np.all(np.isfinite(d)), d
    return d


_FLUID_BACKENDS = ['mpiramS', 'ramgeo', 'ramsurf']


@pytest.mark.parametrize('backend', _FLUID_BACKENDS + ['rams'])
def test_ram_field_shape_follows_the_modal_sum_up_to_a_common_offset(backend):
    """On the analytic suite's 50 Hz / 100 m Pekeris case the difference
    ``RAM − modal sum``, with its own median removed, must stay small at the
    typical cell: a wrong reference speed, starter, grid or absorbing layer
    warps the interference pattern before it moves the mean level.

    Measured over 51 cells, median |d − median d| / p90: mpiramS 0.45 /
    1.12 dB, ramgeo 0.47 / 1.15, ramsurf 0.47 / 1.15 (ramgeo and ramsurf
    return the identical field on a flat surface), rams on the near-fluid
    elastic seabed 0.04 / 0.14. Bounds are 2.1x and 1.8x the fluid values;
    the deterministic fields leave no run-to-run noise to absorb.
    """
    d = _ram_minus_modal_sum(backend, _STANDARD)
    shape = np.abs(d - np.median(d))
    assert np.median(shape) < 1.0, (
        f"{backend}: median |d − median| = {np.median(shape):.2f} dB\n{d}")
    assert np.percentile(shape, 90) < 2.0, (
        f"{backend}: p90 |d − median| = {np.percentile(shape, 90):.2f} dB\n{d}")


@pytest.mark.parametrize('backend', _FLUID_BACKENDS)
def test_ram_absolute_level_matches_the_modal_sum_on_a_well_resolved_case(backend):
    """At 100 Hz in 100 m the PE's own method error is small enough to pin
    RAM's absolute level to the analytic modal sum: measured median |Δ| /
    p90 over 51 cells — mpiramS 0.26 / 0.76 dB, ramgeo 0.24 / 0.87,
    ramsurf 0.24 / 0.87, with means +0.12 / −0.01 / −0.01 dB. Bounds are 3x
    and 1.7x those. A factor-2 power error is 3.01 dB in every cell and
    fails the median by 3.8x.

    The frequency / depth sweep that chose this case (all on the auto grid,
    median |Δ| for mpiramS): 50 Hz/100 m 0.51, 100 Hz/100 m 0.26,
    200 Hz/100 m 3.52, 100 Hz/200 m 1.25, 50 Hz/200 m 0.37 dB.
    """
    d = _ram_minus_modal_sum(backend, _BEST)
    assert np.median(np.abs(d)) < 0.8, (
        f"{backend}: median |dTL| = {np.median(np.abs(d)):.2f} dB\n{d}")
    assert np.percentile(np.abs(d), 90) < 1.5, (
        f"{backend}: p90 |dTL| = {np.percentile(np.abs(d), 90):.2f} dB\n{d}")


def test_rams_absolute_level_matches_the_modal_sum_on_a_near_fluid_seabed():
    """rams (Collins's elastic PE) on the 50 Hz / 100 m case with a 50 m/s
    shear seabed reproduces the FLUID modal sum in absolute dB: measured
    median |Δ| 0.04 dB, p90 0.14, max 0.53 over 51 cells — the λ_s/8 grid
    the shear speed forces (dz = 0.071 m) resolves the water column far
    better than the fluid backends' auto grid, which is why this is the
    tightest RAM pin in the suite. Bounds are 7x and 4x the measurement.
    Shear speeds of 100 and 200 m/s read 0.18 and 1.03 dB median: the
    seabed stops being "near-fluid" and the reference stops being exact, so
    the fixture stays at 50 m/s.
    """
    d = _ram_minus_modal_sum('rams', _STANDARD)
    assert np.median(np.abs(d)) < 0.3, (
        f"rams: median |dTL| = {np.median(np.abs(d)):.3f} dB\n{d}")
    assert np.percentile(np.abs(d), 90) < 0.6, (
        f"rams: p90 |dTL| = {np.percentile(np.abs(d), 90):.3f} dB\n{d}")
