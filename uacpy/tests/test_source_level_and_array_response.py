"""Source level as a field, and what a source array does to the channel.

Two questions a multi-source run raises that transmission loss cannot
answer on its own: *how loud is it actually* (§ level), and *what does
driving several sources together do* (§ array response). The second has two
correct answers — a free-field beam pattern and a modal excitation — and the
tests below pin both, because in a waveguide only the second one sets the
field (Medwin & Clay §11.3.1, "Arrays of sources and receivers in a
waveguide: mode filters").
"""

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field

F0 = 100.0


def _complex_field(value=1.0 + 0.0j, shape=(2, 3)):
    return Field(
        data=np.full(shape, value, dtype=complex),
        coords={'depth': [10.0, 20.0], 'range': [100.0, 200.0, 300.0]},
        model='Test', frequencies=F0, source_depths=[10.0],
        phase_reference='travelling_wave',
    )


# ── source level → an absolute level field ──────────────────────────────


def test_at_source_level_subtracts_the_loss_from_the_source_level():
    """RL = SL - TL, the sonar equation's propagation term. The field keeps
    its grid and gains the level the source was driven at."""
    field = _complex_field(0.01 + 0.0j)          # |p| = 0.01 -> TL = 40 dB
    out = field.at_source_level(180.0)
    np.testing.assert_allclose(np.asarray(out.data), 180.0 - 40.0, atol=1e-9)
    np.testing.assert_array_equal(out.coords['range'], field.coords['range'])
    assert out.metadata['source_level_dB'] == 180.0


def test_a_level_is_its_own_kind_so_it_is_not_labelled_transmission_loss():
    """The colorbar comes from ``kind``; a level that inherited 'pressure'
    would be captioned "TL (dB)" and read backwards on a 1-D cut."""
    from uacpy.visualization.plots._common import _is_loss_view, _value_label
    out = _complex_field(0.01 + 0.0j).at_source_level(180.0)
    assert out.kind == 'level'
    assert out.unit == 'dB'
    assert 'TL' not in _value_label(out, 'dB')
    assert 'Level' in _value_label(out, 'dB')
    # More of a level is louder, so a cut through it reads upward.
    assert _is_loss_view(out, 'dB') is False


def test_at_source_level_reads_a_real_db_field_as_the_loss_it_already_is():
    """A dB-only result (Kraken INCOHERENT_TL, OAST, or an incoherent
    superposition) is already the loss, so it is subtracted, not re-derived."""
    db = _complex_field(0.01 + 0.0j).to_dB()
    np.testing.assert_allclose(np.asarray(db.data), 40.0, atol=1e-9)
    out = db.at_source_level(180.0)
    np.testing.assert_allclose(np.asarray(out.data), 140.0, atol=1e-9)


def test_at_source_level_refuses_a_time_domain_trace():
    """A trace is linear pressure, not a loss; there is nothing to subtract."""
    trace = Field(data=np.ones((2, 4)),
                  coords={'depth': [10.0, 20.0], 'time': [0.0, 1.0, 2.0, 3.0]},
                  model='Test', frequencies=F0, source_depths=[10.0],
                  phase_reference='time_domain_native')
    with pytest.raises(ConfigurationError, match='time-domain'):
        trace.at_source_level(180.0)


def test_a_superposed_total_carries_its_source_level_through():
    """The multi-source level map: combine the sources, then say how hard
    they were driven."""
    from uacpy.core.results import ResultStack
    stack = ResultStack([_complex_field(0.01), _complex_field(0.01)],
                        [10.0, 20.0])
    out = stack.superpose(coherent=False).at_source_level(180.0)
    assert out.kind == 'level'
    # Two equal incoherent sources: 10*log10(2) louder than one.
    np.testing.assert_allclose(np.asarray(out.data),
                               140.0 + 10.0 * np.log10(2.0), atol=1e-9)


# ── the array: free-field pattern and modal excitation ──────────────────


def test_the_array_factor_matches_a_brute_force_free_field_sum():
    """The product theorem (Balanis 6-5) says the array's far field is the
    element field times ``AF(theta)``. Pin that against the actual sum over
    elements at a large radius — which is what fixes the sign convention
    against uacpy's own ``exp(-ikr)`` carrier, rather than a derivation."""
    c, f = 1500.0, 50.0
    k = 2.0 * np.pi * f / c
    zs = np.array([30.0, 40.0, 50.0])
    w = np.array([1.0, 0.7 + 0.3j, -0.4])
    src = uacpy.Source(depths=zs, frequencies=f, weights=w)

    angles = np.linspace(-60.0, 60.0, 25)
    af = src.array_factor(angles, frequency=f, sound_speed=c)

    z0 = zs.mean()
    R = 4.0e5                                    # far field: R >> 2L^2/lambda
    brute = []
    for th in np.deg2rad(angles):
        x, z = R * np.cos(th), z0 + R * np.sin(th)
        rn = np.sqrt(x ** 2 + (z - zs) ** 2)
        brute.append(np.sum(w * np.exp(-1j * k * rn) / rn)
                     * R / np.exp(-1j * k * R))
    brute = np.array(brute)
    assert np.max(np.abs(af - brute)) < 2e-3 * np.max(np.abs(brute)), (
        np.max(np.abs(af - brute)), np.max(np.abs(brute)))


def test_a_steered_array_factor_peaks_where_it_was_steered():
    c, f = 1500.0, 50.0
    k = 2.0 * np.pi * f / c
    zs = np.linspace(20.0, 60.0, 5)
    for steer in (0.0, 15.0, -25.0):
        w = np.exp(-1j * k * (zs - zs.mean()) * np.sin(np.deg2rad(steer)))
        src = uacpy.Source(depths=zs, frequencies=f, weights=w)
        angles = np.linspace(-80.0, 80.0, 641)
        af = np.abs(src.array_factor(angles, frequency=f, sound_speed=c))
        assert abs(angles[int(np.argmax(af))] - steer) < 1.0, steer


def test_a_single_source_has_a_flat_array_factor():
    """One element is not an array: its factor is its own weight at every
    angle, which is the product theorem's element-only case."""
    src = uacpy.Source(depths=50.0, frequencies=100.0, weights=2.0)
    af = src.array_factor(np.linspace(-90.0, 90.0, 19), frequency=100.0)
    np.testing.assert_allclose(af, 2.0 + 0j, atol=1e-12)


@pytest.mark.requires_binary
def test_the_modal_excitation_is_the_weighted_sum_of_the_mode_shapes():
    """What the array does in a waveguide: it sets each mode's amplitude to
    ``sum_n w_n * phi_m(z_n)``. This is the mode-filter view, and unlike the
    free-field pattern it is exact in the channel."""
    from uacpy.models.kraken import Kraken
    from uacpy.tests.conftest import make_pekeris
    env = make_pekeris(name='excitation', bathymetry=100.0)
    zs = [30.0, 70.0]
    w = np.array([1.0, -1.0])
    src = uacpy.Source(depths=zs, frequencies=F0, weights=w)
    modes = Kraken(verbose=False).compute_modes(env, src)

    got = modes.excitation(src)
    phi = np.array([np.interp(z, modes.depths, modes.phi[:, m].real)
                    + 1j * np.interp(z, modes.depths, modes.phi[:, m].imag)
                    for z in zs for m in range(modes.n_modes)]
                   ).reshape(len(zs), modes.n_modes)
    np.testing.assert_allclose(got, w @ phi, rtol=1e-9, atol=1e-12)
    assert got.shape == (modes.n_modes,)


@pytest.mark.requires_binary
def test_weights_conjugate_to_one_mode_shape_select_that_mode():
    """The property the name 'mode filter' points at (Medwin & Clay
    §11.3.1): drive the elements with a mode's own shape and that mode is
    excited far more strongly than its neighbours."""
    from uacpy.models.kraken import Kraken
    from uacpy.tests.conftest import make_pekeris
    env = make_pekeris(name='filter', bathymetry=100.0)
    zs = np.linspace(10.0, 90.0, 9)
    probe = uacpy.Source(depths=zs, frequencies=F0)
    modes = Kraken(verbose=False).compute_modes(env, probe)
    target = 1                                   # second mode
    shape = np.array([np.interp(z, modes.depths, modes.phi[:, target].real)
                      for z in zs])
    driven = uacpy.Source(depths=zs, frequencies=F0, weights=shape)
    amp = np.abs(modes.excitation(driven))
    assert np.argmax(amp) == target, (amp, target)
    others = np.delete(amp, target)
    assert amp[target] > 3.0 * others.max(), (amp[target], others.max())


@pytest.mark.requires_binary
def test_the_excitation_plot_draws_both_views_against_mode_angle():
    """The figure the two descriptions share an axis on: a stem per mode at
    that mode's grazing angle, and the free-field pattern as a curve. An
    unsteered array puts them on top of each other; steering moves the
    curve and not the stems, which is the whole point."""
    import matplotlib
    matplotlib.use('Agg')
    from uacpy.models.kraken import Kraken
    from uacpy.tests.conftest import make_pekeris
    from uacpy.visualization import plot_mode_excitation
    env = make_pekeris(name='excite-plot', bathymetry=100.0)
    zs = np.linspace(20.0, 60.0, 5)
    src = uacpy.Source(depths=zs, frequencies=F0)
    modes = Kraken(verbose=False).compute_modes(env, src)

    # env= reads the speed at the source depth off the profile; the call
    # refuses to invent one, because the x axis IS an angle against it.
    fig, ax = plot_mode_excitation(modes, src, env=env)
    stems = [c for c in ax.collections] + ax.get_lines()
    assert stems, "nothing drawn"
    assert 'angle' in ax.get_xlabel().lower(), ax.get_xlabel()
    assert 'dB' in ax.get_ylabel(), ax.get_ylabel()
    # The free-field curve is drawn too, and is labelled as free field.
    texts = [t.get_text().lower() for t in ax.get_legend().get_texts()]
    assert any('free' in t for t in texts), texts
    assert any('mode' in t for t in texts), texts
    matplotlib.pyplot.close(fig)


def test_a_source_can_carry_the_level_it_is_driven_at():
    """The drive level belongs to the source, like its depths and its
    weights — not to the call that plots the result."""
    src = uacpy.Source(depths=50.0, frequencies=F0, source_level_dB=180.0)
    assert src.source_level_dB == 180.0
    assert uacpy.Source(depths=50.0, frequencies=F0).source_level_dB is None
    with pytest.raises(ConfigurationError, match='source_level_dB'):
        uacpy.Source(depths=50.0, frequencies=F0, source_level_dB=np.nan)


def test_at_source_level_defaults_to_the_level_the_source_carried():
    """A run made with a levelled Source already knows how loud it was, so
    the level view needs no argument repeated at the call site."""
    field = _complex_field(0.01 + 0.0j)
    field.metadata['source_level_dB'] = 180.0
    np.testing.assert_allclose(np.asarray(field.at_source_level().data),
                               140.0, atol=1e-9)
    # An explicit argument still wins over the stamp.
    np.testing.assert_allclose(
        np.asarray(field.at_source_level(200.0).data), 160.0, atol=1e-9)


def test_at_source_level_says_so_when_no_level_is_available():
    with pytest.raises(ConfigurationError, match='source_level_dB'):
        _complex_field(0.01 + 0.0j).at_source_level()


@pytest.mark.requires_binary
def test_a_run_stamps_the_sources_level_onto_its_result():
    """End to end: give the Source a level, run, ask for the level map."""
    from uacpy.models.kraken import Kraken
    from uacpy.tests.conftest import make_pekeris
    env = make_pekeris(name='sl', bathymetry=100.0)
    src = uacpy.Source(depths=50.0, frequencies=F0, source_level_dB=180.0)
    rcv = uacpy.Receiver(depths=[30.0, 70.0], ranges=[1000.0, 2000.0])
    field = Kraken(verbose=False).run(env, src, rcv)
    assert field.metadata['source_level_dB'] == 180.0
    level = field.at_source_level()
    np.testing.assert_allclose(np.asarray(level.data),
                               180.0 - np.asarray(field.dB), atol=1e-9)
    assert level.kind == 'level'


# ── what a level is, and what it is not ─────────────────────────────────


def test_at_source_level_refuses_a_field_that_is_already_a_level():
    """``SL - (SL - TL)`` is the transmission loss again, and stamping that
    'level' would caption a loss as a level and read it upward. The second
    call has to refuse rather than quietly invert."""
    level = _complex_field(0.01 + 0.0j).at_source_level(180.0)
    with pytest.raises(ConfigurationError, match='already a level'):
        level.at_source_level(200.0)
    with pytest.raises(ConfigurationError, match='already a level'):
        level.at_source_level()          # the stamp makes it argument-free


@pytest.mark.parametrize('kind', ['difference', 'ambiguity', 'signal_excess'])
def test_at_source_level_refuses_a_dB_quantity_that_is_not_a_loss(kind):
    """Only a loss can be subtracted from a source level. A residual, a
    normalised ambiguity power and a signal excess are all dB and none of
    them is a propagation loss."""
    f = _complex_field(0.01 + 0.0j).to_dB()
    f.metadata['kind'] = kind
    with pytest.raises(ConfigurationError, match='not a transmission loss'):
        f.at_source_level(180.0)


def test_an_incoherent_sum_of_levels_adds_them_as_levels():
    """A level is not a loss: two mutually incoherent 120 dB sources make
    123.01 dB, not 116.99. Reading the stored numbers with the loss
    convention inverts the direction."""
    from uacpy.core.results import ResultStack
    slabs = []
    for z in (10.0, 20.0):
        f = _complex_field(0.01 + 0.0j).at_source_level(180.0)   # 140 dB
        f.source_depths = np.array([z])
        slabs.append(f)
    out = ResultStack(slabs, [10.0, 20.0]).superpose(coherent=False)
    assert out.kind == 'level'
    np.testing.assert_allclose(np.asarray(out.data),
                               140.0 + 10.0 * np.log10(2.0), atol=1e-9)


def test_an_incoherent_sum_checks_the_grids_like_a_coherent_one():
    """Cells sampled at different ranges must not be added and labelled
    with one of the two axes."""
    from uacpy.core.results import ResultStack
    a = _complex_field(0.01 + 0.0j)
    b = Field(data=np.full((2, 3), 0.01 + 0.0j),
              coords={'depth': [10.0, 20.0], 'range': [100.0, 200.0, 999.0]},
              model='Test', frequencies=F0, source_depths=[20.0],
              phase_reference='travelling_wave')
    with pytest.raises(ConfigurationError, match='different grid'):
        ResultStack([a, b], [10.0, 20.0]).superpose(coherent=False)


def test_an_incoherent_sum_floors_a_null_like_every_other_dB_view():
    """A zero cell reads the package's dB floor, not ``inf`` — an infinity
    propagates into colour scales and into every statistic over the field."""
    from uacpy.core.results import ResultStack
    a = _complex_field(0.0 + 0.0j)
    b = Field(data=np.zeros((2, 3), dtype=complex),
              coords=dict(a.coords), model='Test', frequencies=F0,
              source_depths=[20.0], phase_reference='travelling_wave')
    out = ResultStack([a, b], [10.0, 20.0]).superpose(coherent=False)
    assert np.all(np.isfinite(np.asarray(out.data))), np.asarray(out.data)
    np.testing.assert_allclose(np.asarray(out.data),
                               np.asarray(a.dB), atol=1e-6)


# ── the tabulation grid keeps the endpoints KRAKEN demands ──────────────


def test_the_merged_grid_keeps_its_endpoints_for_the_coupled_mode_check():
    """``EvaluateCMMod.f90:312`` stops the run unless ``z(1) == depthT`` and
    ``z(NR) == depthB`` exactly, which ``io/oalib_writer`` already documents.
    A receiver a micron off an endpoint must not displace it."""
    from uacpy.models.kraken import _merge_depths
    grid = np.linspace(0.0, 100.0, 151)
    merged = _merge_depths(grid, [1e-06, 30.0, 60.0, 100.0 - 1e-06], 100.0)
    assert merged[0] == 0.0, merged[:3]
    assert merged[-1] == 100.0, merged[-3:]


def test_the_merged_grid_separates_every_pair_it_emits():
    """The spacing the comment promises is one MergeVectors cannot collapse,
    and that has to hold between two receivers as well as between a receiver
    and the grid — two legal depths 2 um apart are inside its tolerance."""
    from uacpy.models.kraken import _merge_depths, _MERGE_VECTORS_TOL_M
    merged = _merge_depths(np.linspace(0.0, 100.0, 101),
                           [50.000000, 50.000002, 75.0], 100.0)
    assert np.all(np.diff(merged) > _MERGE_VECTORS_TOL_M), np.diff(merged).min()


@pytest.mark.requires_binary
def test_a_coupled_run_survives_a_receiver_next_to_the_surface():
    """The regression the endpoint rule prevents: this worked before the
    receiver depths joined the tabulation grid, and must still."""
    from uacpy.models.kraken import Kraken
    env = uacpy.Environment(
        name='wedge', bathymetry=[(0.0, 100.0), (4000.0, 70.0)], ssp=1500.0,
        bottom=uacpy.core.bottom.Bottom.from_halfspace(
            uacpy.core.BoundaryProperties(acoustic_type='half-space',
                                          sound_speed=1600.0, density=1.8,
                                          attenuation=0.2)))
    src = uacpy.Source(depths=50.0, frequencies=F0)
    rcv = uacpy.Receiver(depths=[1e-06, 30.0, 60.0], ranges=[1000.0, 3000.0])
    field = Kraken(verbose=False, mode_coupling='coupled').run(env, src, rcv)
    assert np.isfinite(np.asarray(field.dB)).any()


# ── the array-factor curve covers the angles the array radiates into ────


@pytest.mark.requires_binary
def test_the_excitation_plot_shows_an_upward_steered_main_lobe():
    """F15: the curve used to be drawn over [0, 90] only, so an array
    steered upward had its main lobe off-plot and a sidelobe 12 dB down
    normalised to 0 dB. A mode is a standing wave and responds to both
    halves, so the comparable free-field quantity is folded over ±θ."""
    import matplotlib
    matplotlib.use('Agg')
    from uacpy.models.kraken import Kraken
    from uacpy.tests.conftest import make_pekeris
    from uacpy.visualization import plot_mode_excitation
    env = make_pekeris(name='steered', bathymetry=100.0)
    zs = np.linspace(20.0, 60.0, 5)
    c, f = 1500.0, 200.0
    k = 2.0 * np.pi * f / c
    probe = uacpy.Source(depths=zs, frequencies=f)
    modes = Kraken(verbose=False).compute_modes(env, probe)
    w = np.exp(-1j * k * (zs - zs.mean()) * np.sin(np.deg2rad(-30.0)))
    src = uacpy.Source(depths=zs, frequencies=f, weights=w)
    fig, ax = plot_mode_excitation(modes, src, sound_speed=c)
    curve = [ln for ln in ax.get_lines() if ln.get_label().startswith('array')]
    assert curve, [ln.get_label() for ln in ax.get_lines()]
    x, y = curve[0].get_xdata(), curve[0].get_ydata()
    # Normalising to the curve's own maximum always reaches 0 dB, so that
    # proves nothing. What the bug moved is WHERE the peak sits: folded over
    # +/-theta, an array steered to -30 deg peaks at +30 deg. Drawn over
    # [0, 90] unfolded it peaked at ~+67 deg, a sidelobe 12 dB down.
    assert abs(x[int(np.nanargmax(y))] - 30.0) < 2.0, x[int(np.nanargmax(y))]
    # ... and the true main lobe is the reference, so the stems are compared
    # against a curve whose 0 dB is the array's real peak.
    af = np.abs(src.array_factor(np.array([-30.0]), frequency=f,
                                 sound_speed=c))[0]
    assert abs(af - len(zs)) < 1e-6, af
    matplotlib.pyplot.close(fig)


def test_a_multi_source_total_is_not_captioned_transmission_loss():
    """A superposed field keeps TL's reference — one unit source at 1 m —
    so the array's gain sits inside the number and it can go NEGATIVE.
    Calling that "TL" is what JKPS §1.3.4 and Porter's own guide define
    against a single source, so the caption has to say what it really is."""
    from uacpy.core.results import ResultStack
    from uacpy.visualization.plots._common import _value_label
    slabs = [_complex_field(0.5 + 0.0j) for _ in range(2)]
    for i, s in enumerate(slabs):
        s.source_depths = np.array([10.0 * (i + 1)])
    total = ResultStack(slabs, [10.0, 20.0]).superpose()
    label = _value_label(total, 'dB')
    assert 'TL' not in label, label
    assert 'total' in label.lower(), label
    # A single-source field is untouched: it really is transmission loss.
    assert _value_label(_complex_field(0.5 + 0.0j), 'dB') == 'TL (dB)'


def test_a_total_that_is_louder_than_one_unit_source_reads_negative():
    """The number the caption has to be honest about: enough in-phase
    sources drive the 'loss' below zero, which no transmission loss does."""
    from uacpy.core.results import ResultStack
    slabs = [_complex_field(0.5 + 0.0j) for _ in range(8)]
    for i, s in enumerate(slabs):
        s.source_depths = np.array([10.0 * (i + 1)])
    total = ResultStack(slabs, np.arange(1, 9) * 10.0).superpose()
    assert np.nanmin(np.asarray(total.dB)) < 0.0, np.nanmin(total.dB)


@pytest.mark.requires_binary
def test_a_weighted_source_on_a_db_mode_yields_a_stack_to_combine():
    """A dB-only mode cannot sum coherently, but it CAN sum in intensity —
    that is the whole point of the incoherent path. Refusing the weights
    before the stack exists closed the one route to 'N mutually incoherent
    sources at these levels'."""
    from uacpy.models.kraken import Kraken
    from uacpy.core.results import ResultStack
    from uacpy.tests.conftest import make_pekeris
    env = make_pekeris(name='db-weights', bathymetry=100.0)
    rcv = uacpy.Receiver(depths=[30.0, 70.0], ranges=[1000.0, 2000.0])
    src = uacpy.Source(depths=[30.0, 70.0], frequencies=F0, weights=[1.0, 2.0])
    stack = Kraken(verbose=False).run(env, src, rcv,
                                      run_mode=uacpy.RunMode.INCOHERENT_TL)
    assert isinstance(stack, ResultStack)
    total = stack.superpose(coherent=False)        # uses the stamped weights
    assert total.unit == 'dB'
    # A coherent sum of the same slabs is still undefined.
    with pytest.raises(ConfigurationError, match='coherent sum'):
        stack.superpose()
    # A COMPLEX weight is unusable either way and is still refused up front.
    bad = uacpy.Source(depths=[30.0, 70.0], frequencies=F0, weights=[1.0, 1j])
    with pytest.raises(ConfigurationError):
        Kraken(verbose=False).run(env, bad, rcv,
                                  run_mode=uacpy.RunMode.INCOHERENT_TL)


@pytest.mark.requires_binary
def test_one_launch_warns_once_about_the_source_axis():
    """Kraken masks each slab of a native stack separately; the warning
    belongs to the run, not to the slab count."""
    import warnings as _w
    from uacpy.models.kraken import Kraken
    from uacpy.tests.conftest import make_pekeris
    env = make_pekeris(name='r0', bathymetry=100.0)
    rcv = uacpy.Receiver(depths=[30.0, 70.0], ranges=[0.0, 1000.0])
    src = uacpy.Source(depths=[20.0, 40.0, 60.0, 80.0], frequencies=F0)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter('always')
        Kraken(verbose=False).run(env, src, rcv)
    hits = [x for x in caught if 'r = 0' in str(x.message)]
    assert len(hits) == 1, [str(x.message)[:60] for x in hits]


# ── the product theorem: P = f · A ──────────────────────────────────────


def _shaped():
    """A directional element: 0 dB broadside, rolling off to -20 dB."""
    return np.array([[-90.0, -20.0], [-45.0, -6.0], [0.0, 0.0],
                     [45.0, -6.0], [90.0, -20.0]])


def test_the_array_beam_pattern_is_the_element_pattern_times_the_factor():
    """Butler & Sherman §7.1.1: ``P(θ) = f(θ)·A(θ)`` — the array beam
    pattern is the element beam pattern times the beam pattern of the array
    of point sources. uacpy's ``beam_pattern`` IS ``f``, and ``array_factor``
    is ``A``, so the two must multiply to the one a user actually radiates."""
    zs = np.linspace(20.0, 60.0, 5)
    src = uacpy.Source(depths=zs, frequencies=200.0, beam_pattern=_shaped())
    angles = np.linspace(-80.0, 80.0, 161)
    P = src.array_beam_pattern(angles, sound_speed=1500.0)
    A = src.array_factor(angles, sound_speed=1500.0)
    # f interpolated the way the engines read it: dB -> amplitude FIRST.
    f = np.interp(angles, _shaped()[:, 0], 10.0 ** (_shaped()[:, 1] / 20.0))
    np.testing.assert_allclose(P, f * A, rtol=1e-12)


def test_the_array_factor_alone_is_the_isotropic_element_case():
    """``A`` is defined for point sources — Balanis: it "does not depend on
    the directional characteristics of the radiating elements". So a shaped
    source and an omni one share one array factor, and differ in ``P``."""
    zs = np.linspace(20.0, 60.0, 5)
    angles = np.linspace(-80.0, 80.0, 81)
    shaped = uacpy.Source(depths=zs, frequencies=200.0,
                          beam_pattern=_shaped())
    omni = uacpy.Source(depths=zs, frequencies=200.0)
    np.testing.assert_allclose(shaped.array_factor(angles, sound_speed=1500.0),
                               omni.array_factor(angles, sound_speed=1500.0),
                               rtol=1e-12)
    assert not np.allclose(
        np.abs(shaped.array_beam_pattern(angles, sound_speed=1500.0)),
        np.abs(omni.array_beam_pattern(angles, sound_speed=1500.0)))


def test_an_omnidirectional_source_radiates_its_bare_array_factor():
    """With no element pattern there is nothing to multiply by: P == A."""
    src = uacpy.Source(depths=[20.0, 40.0], frequencies=200.0)
    angles = np.linspace(-60.0, 60.0, 25)
    np.testing.assert_allclose(src.array_beam_pattern(angles, sound_speed=1500.0),
                               src.array_factor(angles, sound_speed=1500.0),
                               rtol=1e-12)


@pytest.mark.requires_binary
def test_the_modal_excitation_carries_the_element_pattern_the_engine_applies():
    """``field.f90`` shades the modal excitation by ``S(θₘ)`` before summing
    (``C = C * REAL(S)``), and every element shares one pattern and one mode
    angle, so ``S`` factors out of the array sum: the true excitation is
    ``S(θₘ)·Σₙ wₙ·φₘ(zₙ)``. Omitting it made a shaded array look omni."""
    from uacpy.models.kraken import Kraken
    from uacpy.tests.conftest import make_pekeris
    env = make_pekeris(name='shaded', bathymetry=100.0)
    zs = np.linspace(20.0, 60.0, 5)
    bare = uacpy.Source(depths=zs, frequencies=200.0)
    shaped = uacpy.Source(depths=zs, frequencies=200.0,
                          beam_pattern=_shaped())
    modes = Kraken(verbose=False).compute_modes(env, bare)

    plain = modes.excitation(bare)
    shaded = modes.excitation(shaped, sound_speed=1500.0)
    angles = np.degrees(np.arccos(np.clip(
        1500.0 / np.asarray(modes.compute_phase_speeds(), dtype=float),
        -1.0, 1.0)))
    f = np.interp(angles, _shaped()[:, 0], 10.0 ** (_shaped()[:, 1] / 20.0))
    np.testing.assert_allclose(shaded, f * plain, rtol=1e-9, atol=1e-12)
    # A shaded source whose pattern is never asked for is a silent omission.
    with pytest.raises(ConfigurationError, match='sound_speed'):
        modes.excitation(shaped)


def test_each_panel_of_a_source_depth_grid_marks_its_own_source():
    """A panel grid over ``source_depth`` is 'one source at a time', so each
    panel draws the source it belongs to — not the whole array, which would
    say the opposite, and not nothing, which leaves the reader to hand-roll
    a marker (example 04 did exactly that)."""
    import matplotlib
    matplotlib.use('Agg')
    from uacpy.core.results import ResultStack
    depths = [10.0, 20.0, 30.0]
    slabs = []
    for z in depths:
        f = _complex_field(0.01 + 0.0j)
        f.source_depths = np.array([z])
        slabs.append(f)
    fig, axes = ResultStack(slabs, depths).plot()
    flat = np.asarray(axes).ravel()
    for z, ax in zip(depths, flat):
        marked = [ln.get_ydata()[0] for ln in ax.get_lines()
                  if len(ln.get_ydata()) == 1]
        assert any(abs(y - z) < 1e-9 for y in marked), (z, marked)
        # ... and only its own.
        assert not any(abs(y - other) < 1e-9
                       for y in marked for other in depths if other != z), (
            z, marked)
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize('plotter,kind,data', [
    ('plot_detection_probability', 'probability_of_detection', 0.6),
    ('plot_signal_excess', 'signal_excess', 3.0),
])
def test_the_sonar_heatmaps_draw_the_geometry_they_are_about(plotter, kind,
                                                             data):
    """A detection map answers 'would this array hear that target' — so it
    has to be able to show where the array and the target are. Both sonar
    panels already take ``env`` for the seafloor; the geometry belongs on
    the same footing, as it is on ``plot_field``."""
    import matplotlib
    matplotlib.use('Agg')
    from uacpy import visualization
    fn = getattr(visualization, plotter)
    field = Field(
        data=np.full((3, 4), data, dtype=float),
        coords={'depth': [10.0, 50.0, 90.0],
                'range': [500.0, 1000.0, 1500.0, 2000.0]},
        model='Test', frequencies=F0, source_depths=[30.0],
        metadata={'kind': kind},
    )
    src = uacpy.Source(depths=30.0, frequencies=F0)
    rcv = uacpy.Receiver(depths=[40.0, 60.0], ranges=[1000.0, 2000.0])
    fig, ax = fn(field, source=src, receiver=rcv)
    marked = [ln.get_ydata()[0] for ln in ax.get_lines()
              if len(np.atleast_1d(ln.get_ydata())) == 1]
    assert any(abs(y - 30.0) < 1e-9 for y in marked), marked
    matplotlib.pyplot.close(fig)
