"""Independent, hand-written tests for the matched-field processing layer.

The decisive correctness oracle is physical: the Python modal-sum replica
(:func:`synthesize_replica`) must reproduce ``field.exe``'s complex pressure
(``Kraken.run`` COHERENT_TL) up to a single global complex scalar. The
processor tests inject a known source and check the ambiguity surface peaks
at the true location.
"""

import inspect

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
        modes, source_depth=25.0, ranges=rcv.ranges, array_depths=rcv.depths
    )
    P = field.data  # (n_depth, n_range)
    assert syn.shape == P.shape
    corr = abs(np.vdot(syn, P)) / (np.linalg.norm(syn) * np.linalg.norm(P))
    # The magnitude of a normalised correlation is invariant to the global
    # complex scalar the replica omits, so it isolates the shape. What is left
    # is the far-field (asymptotic Hankel) approximation both sides make, which
    # is worth well under 1e-3 here: the nearest range is 200 m, i.e. k*r ≈ 13.
    assert corr > 0.999, f"replica vs field.exe normalized corr = {corr}"


def test_replica_vector_has_one_entry_per_array_element(pekeris):
    modes = pekeris["modes"]
    z_arr = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
    e = synthesize_replica(modes, source_depth=40.0, ranges=2000.0, array_depths=z_arr)
    assert e.shape == (z_arr.size,)
    assert np.iscomplexobj(e)


def test_replica_obeys_source_receiver_reciprocity(pekeris):
    # The modal sum is symmetric in source/receiver depth: the pressure at z_b
    # from a source at z_a equals the pressure at z_a from a source at z_b.
    modes = pekeris["modes"]
    p_ab = synthesize_replica(modes, source_depth=20.0, ranges=2500.0, array_depths=70.0)
    p_ba = synthesize_replica(modes, source_depth=70.0, ranges=2500.0, array_depths=20.0)
    assert p_ab == pytest.approx(p_ba, rel=1e-12)


def test_synthesize_replica_rejects_nonpositive_range(pekeris):
    modes = pekeris["modes"]
    with pytest.raises(ConfigurationError, match='ranges must be > 0'):
        synthesize_replica(modes, source_depth=25.0, ranges=0.0, array_depths=50.0)


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
    syn = synthesize_replica(modes, source_depth=30.0, ranges=rcv.ranges, array_depths=rcv.depths)
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
    bank = replica_bank(modes, array_depths=z_arr, candidate_depths=cand_z,
                        candidate_ranges=cand_r)

    iz, ir = 18, 25                            # true source grid cell
    d = bank.replicas[0, iz, ir]               # noise-free data = that replica
    K = csdm(d[:, None])
    surf = bartlett(K, bank)
    assert surf.shape == (cand_z.size, cand_r.size)
    peak = np.unravel_index(np.nanargmax(surf.data), surf.shape)
    assert peak == (iz, ir)


def test_the_ambiguity_field_reads_the_estimate_off_its_coordinates(pekeris):
    """The surface is a ``kind='ambiguity'`` Field in dB re max, so
    ``.max()`` is the estimate with its coordinates, and its ``reference``
    is the peak power the dB values are relative to: the linear surface of
    :func:`uacpy.acoustic_signal.bartlett` is recovered from the result
    alone."""
    from uacpy.acoustic_signal import bartlett as power
    from uacpy.core.results import ambiguity_field
    z_arr = np.linspace(5, 95, 16)
    cand_z = np.linspace(5, 95, 46)
    cand_r = np.linspace(500, 5000, 60)
    bank = replica_bank(pekeris["modes"], array_depths=z_arr,
                        candidate_depths=cand_z, candidate_ranges=cand_r)
    K = csdm(bank.replicas[0, 18, 25][:, None])
    field = bartlett(K, bank)
    assert field.kind == 'ambiguity' and field.unit == 'dB'
    assert field.model == 'Kraken' and field.reference_unit == '1'
    assert float(np.nanmax(field.data)) == 0.0
    linear = power(K, bank.replicas[0], normalize='trace')
    np.testing.assert_allclose(field.reference * 10 ** (field.data / 10),
                               linear, rtol=1e-12)
    best = field.max()
    assert best.pinned == {'depth': cand_z[18], 'range': cand_r[25]}
    # The surface is indexed by the candidates in order: range before depth
    # is refused.
    with pytest.raises(ConfigurationError, match='candidates in order'):
        ambiguity_field(linear, {'range': cand_r, 'depth': cand_z},
                        reference_unit='1')
    with pytest.raises(ConfigurationError, match='no finite positive peak'):
        ambiguity_field(np.zeros_like(linear),
                        {'depth': cand_z, 'range': cand_r},
                        reference_unit='1')


def test_the_processors_take_the_covariance_by_name(pekeris):
    """``K`` is the formulas' symbol; the argument is named for what it is,
    as the array-processing surface names its own."""
    import inspect
    for fn in (bartlett, mvdr):
        params = list(inspect.signature(fn).parameters)
        assert params[:2] == ['covariance', 'replicas']
    bank = replica_bank(pekeris["modes"], array_depths=np.linspace(5, 95, 16),
                        candidate_depths=np.linspace(5, 95, 8),
                        candidate_ranges=np.linspace(500, 5000, 9))
    k = csdm(bank.replicas[0, 3, 4][:, None])
    np.testing.assert_array_equal(bartlett(covariance=k, replicas=bank).data,
                                  bartlett(k, bank).data)


def test_bartlett_is_normalized_to_unit_at_perfect_match(pekeris):
    modes = pekeris["modes"]
    z_arr = np.linspace(5, 95, 16)
    cand_z = np.linspace(5, 95, 20)
    cand_r = np.linspace(500, 5000, 20)
    bank = replica_bank(modes, array_depths=z_arr, candidate_depths=cand_z,
                        candidate_ranges=cand_r)
    d = bank.replicas[0, 7, 11]
    field = bartlett(csdm(d[:, None]), bank)
    surf = field.reference * 10 ** (field.data / 10)
    assert surf.min() >= -1e-9 and surf.max() <= 1.0 + 1e-9
    assert surf[7, 11] == pytest.approx(1.0, abs=1e-6)
    assert field.reference == pytest.approx(1.0, abs=1e-6)


# ── MVDR processor ──────────────────────────────────────────────────────

def test_mvdr_peaks_at_true_source(pekeris):
    modes = pekeris["modes"]
    z_arr = np.linspace(5, 95, 16)
    cand_z = np.linspace(5, 95, 40)
    cand_r = np.linspace(500, 5000, 50)
    bank = replica_bank(modes, array_depths=z_arr, candidate_depths=cand_z,
                        candidate_ranges=cand_r)

    iz, ir = 22, 30
    d = bank.replicas[0, iz, ir]
    K = csdm(d[:, None])
    # Loading is needed for a rank-1 K.
    surf = mvdr(K, bank, diagonal_loading=1e-2)
    peak = np.unravel_index(np.nanargmax(surf.data), surf.shape)
    assert peak == (iz, ir)


@pytest.mark.parametrize("processor", ["bartlett", "mvdr"])
def test_localizes_offgrid_source_from_noisy_snapshots(pekeris, processor):
    # Truth sits *between* grid nodes; recover it to the nearest cell at 10 dB
    # SNR from 50 snapshots with random source phase + sensor noise.
    modes = pekeris["modes"]
    z_arr = np.linspace(5, 95, 16)
    true_z, true_r = 62.0, 3200.0
    e_true = synthesize_replica(modes, source_depth=true_z, ranges=true_r, array_depths=z_arr)

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
    bank = replica_bank(modes, array_depths=z_arr, candidate_depths=cand_z,
                        candidate_ranges=cand_r)
    surf = (bartlett(K, bank) if processor == "bartlett"
            else mvdr(K, bank, diagonal_loading=1e-2))

    iz, ir = np.unravel_index(np.nanargmax(surf.data), surf.shape)
    # nearest grid node to truth
    assert abs(cand_z[iz] - true_z) <= (cand_z[1] - cand_z[0])
    assert abs(cand_r[ir] - true_r) <= (cand_r[1] - cand_r[0])


def test_replica_decays_under_either_imag_k_sign():
    """Raw Kraken carries Im(k) < 0; ``Modes.with_attenuation`` builds
    Im(k) > 0. A passive medium can only attenuate, so the replica must decay
    with range under BOTH.

    ``exp(i k r)`` decays only for Im(k) > 0, so the magnitude has to be taken
    rather than the value as given. Over this 19 km span both signs must land
    on ``exp(-|Im k| dr)`` = 3.3e-3; the other branch grows by ~300x, and a
    replica that grows with range moves the MFP peak to the far edge of the
    candidate grid.
    """
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
        p = synthesize_replica(modes, source_depth=25.0, ranges=r, array_depths=np.array([50.0]))
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
        p = synthesize_replica(modes, source_depth=25.0, ranges=r, array_depths=z_r)
        amps.append(np.abs(np.asarray(p).ravel()[0]))
    assert amps[0] / amps[1] == pytest.approx(np.sqrt(k[1] / k[0]), rel=1e-9)


class TestTheTwoBeamformingSurfacesDifferAsDocumented:
    """``sonar.bartlett`` / ``sonar.mvdr`` are the array functions
    :func:`uacpy.acoustic_signal.bartlett` / :func:`~uacpy.acoustic_signal.mvdr`
    over a Replicas bank, and differ from them in two stated ways: the
    normalisation (trace / max against none) and the MVDR loading default.
    The differences are pinned so the ``matched_field`` module docstring
    cannot drift from them, and so the loading defaults are not "tidied"
    into agreement — that would silently change every existing call on one
    side."""

    N, L = 8, 50

    def _snapshots(self):
        rng = np.random.default_rng(0)
        return (rng.standard_normal((self.N, self.L))
                + 1j * rng.standard_normal((self.N, self.L)))

    def _bank(self, rows):
        """A one-frequency Replicas holding ``rows`` (n, N), one per
        candidate range."""
        from uacpy.core.results import Replicas
        return Replicas(replicas=rows[None],
                        candidates={'range': np.arange(1.0, rows.shape[0] + 1)})

    @staticmethod
    def _linear(field):
        return field.reference * 10 ** (field.data / 10)

    def test_the_two_covariance_names_are_one_estimate(self):
        from uacpy.acoustic_signal.beamforming import sample_covariance
        d = self._snapshots()
        np.testing.assert_array_equal(csdm(d), sample_covariance(d))

    def test_the_bartlett_surfaces_differ_by_the_trace(self):
        from uacpy.acoustic_signal import bartlett as power
        d = self._snapshots()
        K = csdm(d)
        rng = np.random.default_rng(3)
        rows = rng.standard_normal((3, self.N)) + 1j * rng.standard_normal(
            (3, self.N))
        surface = self._linear(bartlett(K, self._bank(rows)))
        spectrum = power(K, rows)
        np.testing.assert_allclose(spectrum, surface
                                   * float(np.trace(K).real), rtol=1e-10)

    def test_the_mvdr_surfaces_differ_by_a_single_scale(self):
        from uacpy.acoustic_signal import mvdr as power
        d = self._snapshots()
        K = csdm(d)
        rng = np.random.default_rng(4)
        rows = rng.standard_normal((3, self.N)) + 1j * rng.standard_normal(
            (3, self.N))
        # Same loading on both sides, so only the output scaling differs.
        surface = self._linear(mvdr(K, self._bank(rows),
                                    diagonal_loading=1e-6))
        spectrum = power(K, rows, diagonal_loading=1e-6)
        ratio = spectrum / surface
        assert np.ptp(ratio) / ratio.mean() < 1e-9
        assert surface.max() == pytest.approx(1.0)

    def test_every_mvdr_surface_names_its_loading_diagonal_loading(self):
        """All three MVDR surfaces spell the diagonal-loading knob
        ``diagonal_loading``, so a value moves between them by keyword. The
        parameter stays positional-or-keyword on ``sonar.mvdr``, so a
        caller may also pass it third and positionally."""
        import inspect
        from uacpy.acoustic_signal import mvdr as power
        from uacpy.core.results import Covariance

        for fn in (mvdr, power, Covariance.mvdr):
            par = inspect.signature(fn).parameters
            assert 'diagonal_loading' in par, f"{fn.__qualname__}: {list(par)}"

        assert (inspect.signature(mvdr).parameters['diagonal_loading'].kind
                is inspect.Parameter.KEYWORD_ONLY)
        K = np.eye(4, dtype=complex)
        bank = self._bank(np.eye(4, dtype=complex)[:2])
        np.testing.assert_allclose(mvdr(K, bank, diagonal_loading=1e-2).data,
                                   mvdr(K, bank, diagonal_loading=1e-2).data)

    def test_the_loading_defaults_are_deliberately_different(self):
        import inspect
        from uacpy.acoustic_signal import mvdr as power
        sonar_default = inspect.signature(
            mvdr).parameters['diagonal_loading'].default
        arrays_default = inspect.signature(
            power).parameters['diagonal_loading'].default
        assert sonar_default == 1e-2
        assert arrays_default == 1e-6
        assert sonar_default / arrays_default == pytest.approx(1e4)


@pytest.mark.parametrize("fn, n_arrays", [(synthesize_replica, 3),
                                          (replica_bank, 3)])
def test_the_depth_and_range_arrays_are_keyword_only(fn, n_arrays):
    """Every argument after ``modes`` is a float array (or a depth), and the
    two functions list them in different orders, so a positional call is
    refused before anything runs rather than swapping them silently."""
    args = [np.linspace(5.0, 95.0, 4)] * n_arrays
    with pytest.raises(TypeError, match="positional argument"):
        fn(None, *args)
    params = inspect.signature(fn).parameters
    assert [p.kind for p in params.values()][1:] == [
        inspect.Parameter.KEYWORD_ONLY] * n_arrays


def test_the_ambiguity_field_names_the_plotters_floor_keyword():
    """The docstring points at ``plot_matched_field``'s dB floor by the
    keyword that function takes."""
    import re
    from uacpy.core.results import ambiguity_field
    from uacpy.visualization.plots import plot_matched_field
    named = re.findall(r"``plot_matched_field``'s ``(\w+)``",
                       ambiguity_field.__doc__)
    assert named == ['dynamic_range_dB']
    assert named[0] in inspect.signature(plot_matched_field).parameters
