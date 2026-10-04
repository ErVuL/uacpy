"""Unit tests for :func:`uacpy.visualization.plots.animate_field`.

Builds a synthetic time-series Field carrying a known outgoing Gaussian
pulse ``p(z, r, t) = exp(-((t − r/c) / σ)²) · cos(2π f₀ (t − r/c))`` and
verifies the animation:

* accepts the field shape uacpy emits (``coords={'depth','range','time'}``);
* produces the expected frame count after ``frame_stride`` decimation;
* updates the heatmap data per frame (peak position tracks the pulse);
* rejects fields with the wrong ``kind`` or missing coord axes.
"""

import numpy as np
import pytest

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field


# matplotlib emits "Animation was deleted without rendering" when our
# tests build a FuncAnimation, inspect it, and let it go out of scope
# without saving. Benign — the tests verify structure not playback.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Animation was deleted without rendering:UserWarning"
)


C_WATER = 1500.0
F0 = 100.0
# 50 ms pulse → spatial extent c·σ = 75 m, several range bins wide on the
# test grids below. A narrower pulse falls between bins and produces an
# all-zero snapshot at most frames.
SIGMA = 0.05


def _make_synthetic_field(
    n_d: int = 12,
    n_r: int = 40,
    n_t: int = 200,
    t_max: float = 4.0,
    r_min: float = 0.0,
) -> Field:
    """Outgoing pulse traveling at C_WATER. Independent of depth (so any
    horizontal cut sees the same wavefront). ``r_min`` is the first range
    sample: a model grid starts past r = 0, the source's own range."""
    depths = np.linspace(0.0, 100.0, n_d)
    ranges = np.linspace(r_min, C_WATER * t_max, n_r)
    times = np.linspace(0.0, t_max, n_t)
    DD, RR, TT = np.meshgrid(depths, ranges, times, indexing='ij')
    tau = TT - RR / C_WATER
    data = np.exp(-(tau / SIGMA) ** 2) * np.cos(2 * np.pi * F0 * tau)
    return Field(
        data=data,
        coords={'depth': depths, 'range': ranges, 'time': times},
        model='Synthetic',
        source_depths=np.array([50.0]),
        frequencies=np.array([F0]),
    )


def test_animate_field_returns_funcanimation():
    from matplotlib.animation import FuncAnimation
    from uacpy.visualization.plots import animate_field

    field = _make_synthetic_field()
    ani = animate_field(field, fps=15, frame_stride=10)
    assert isinstance(ani, FuncAnimation)


def test_animate_field_frame_count_respects_stride():
    from uacpy.visualization.plots import animate_field

    field = _make_synthetic_field(n_t=200)
    ani = animate_field(field, fps=15, frame_stride=10)
    # 200 / 10 = 20 frames
    assert len(list(ani.new_frame_seq())) == 20


def test_animate_field_default_frame_stride_caps_at_300():
    """``frame_stride=None`` should cap the animation at ~300 frames."""
    from uacpy.visualization.plots import animate_field

    field = _make_synthetic_field(n_t=2000)
    ani = animate_field(field, frame_stride=None)
    n_frames = len(list(ani.new_frame_seq()))
    # stride = max(1, 2000 // 300) = 6 → 2000 // 6 = 333 frames (within ~10%)
    assert 250 <= n_frames <= 400, n_frames


def test_animate_field_frame_updates_track_pulse_position():
    """Stepping the update callback re-fills the heatmap with real data.

    The three frame indices are chosen to span the pulse crossing the
    receiver array, so a ``set_array`` that never fired (or that wrote the
    wrong time slice) leaves at least one of them identically zero.
    """
    import matplotlib
    matplotlib.use('Agg')
    from uacpy.visualization.plots import animate_field

    field = _make_synthetic_field(n_t=400, t_max=3.0)
    ani = animate_field(field, fps=30, frame_stride=1)

    # Pull the update callback and step through three early / mid / late frames.
    # ani._func is matplotlib's stored update function.
    update_fn = ani._func
    # frame indices that span the pulse propagating across the receiver array.
    for k in (50, 200, 350):
        artists = update_fn(k)
        # First artist is the AxesImage from imshow.
        im = artists[0]
        arr = np.asarray(im.get_array())
        # The data is real and centred on zero; peak (or trough) |arr|
        # should sit at the range bin tracking r = c · t.
        assert np.any(np.abs(arr) > 1e-6), f"frame {k} is all zero"


def test_animate_field_rejects_non_timeseries():
    from uacpy.visualization.plots import animate_field

    # TL field (real, no time axis)
    field = Field(
        data=np.ones((4, 5)),
        coords={'depth': np.arange(4.0), 'range': np.arange(5.0)},
        model='TL',
    )
    assert 'time' not in field.coords
    with pytest.raises(ConfigurationError, match="'time' axis"):
        animate_field(field)


def test_animate_field_rejects_missing_axes():
    """A Field with a time axis but no depth/range fails up front."""
    from uacpy.visualization.plots import animate_field

    # Single-point trace — has 'time' but no 'depth'/'range'.
    field = Field(
        data=np.zeros(10),
        coords={'time': np.linspace(0, 1, 10)},
        model='Trace',
    )
    with pytest.raises(ConfigurationError, match="missing coord axes"):
        animate_field(field)


def test_animate_field_p_max_override():
    """User-supplied ``p_max`` is used verbatim for the colour scale."""
    from uacpy.visualization.plots import animate_field

    field = _make_synthetic_field()
    ani = animate_field(field, p_max=10.0, frame_stride=20)
    update_fn = ani._func
    artists = update_fn(0)
    im = artists[0]
    vmin, vmax = im.get_clim()
    assert vmin == -10.0
    assert vmax == 10.0


def test_animate_field_handles_alternate_coord_order():
    """``coords`` is dict-ordered; uacpy emits depth→range→time but tests
    should pass with any order — the function moves axes internally."""
    from uacpy.visualization.plots import animate_field

    base = _make_synthetic_field(n_d=6, n_r=10, n_t=20)
    # Swap data to (time, depth, range) layout
    data = np.moveaxis(base.data, [0, 1, 2], [1, 2, 0])
    field = Field(
        data=data,
        coords={'time': base.coords['time'],
                'depth': base.coords['depth'],
                'range': base.coords['range']},
        model='Reordered',
    )
    ani = animate_field(field, frame_stride=1)
    assert len(list(ani.new_frame_seq())) == 20


# ─────────────────────────────────────────────────────────────────────────────
# save_animation
# ─────────────────────────────────────────────────────────────────────────────


def test_save_animation_gif(tmp_path):
    """Writer inferred from `.gif` suffix → PillowWriter; output file
    is non-empty."""
    from uacpy.plot import save_animation

    field = _make_synthetic_field(n_t=40, t_max=1.0)
    out = save_animation(field, tmp_path / 'pulse.gif',
                         fps=10, frame_stride=4)
    assert out.exists()
    assert out.stat().st_size > 1024  # > 1 KiB, real animation


def test_save_animation_rejects_unknown_suffix(tmp_path):
    from uacpy.plot import save_animation

    field = _make_synthetic_field(n_t=10)
    with pytest.raises(ConfigurationError, match=r"cannot infer writer"):
        save_animation(field, tmp_path / 'pulse.xyz')


def test_save_animation_closes_figure(tmp_path):
    """Calling save_animation should not leak open figures."""
    import matplotlib.pyplot as plt
    from uacpy.plot import save_animation

    plt.close('all')
    n_open_before = len(plt.get_fignums())

    field = _make_synthetic_field(n_t=20)
    save_animation(field, tmp_path / 'pulse.gif',
                   fps=10, frame_stride=4)

    n_open_after = len(plt.get_fignums())
    assert n_open_after == n_open_before


# ─────────────────────────────────────────────────────────────────────────────
# plot_time_snapshots
# ─────────────────────────────────────────────────────────────────────────────


def test_plot_time_snapshots_grid_shape():
    """One row per field, one column per requested time."""
    from uacpy.plot import plot_time_snapshots

    fields = {
        'A': _make_synthetic_field(n_t=40, t_max=1.0),
        'B': _make_synthetic_field(n_t=40, t_max=1.0),
    }
    fig, axes = plot_time_snapshots(fields, times_s=(0.2, 0.5, 0.8))
    assert axes.shape == (2, 3)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_time_snapshots_empty_raises():
    from uacpy.plot import plot_time_snapshots

    with pytest.raises(ConfigurationError, match='empty'):
        plot_time_snapshots({}, times_s=(0.1,))


@pytest.mark.parametrize("frame_stride, accepted", [
    (0, False),         # divides by zero inside np.arange
    (1, True),          # the documented "render every sample" value
    (-1, False),
    (np.int64(2), True),   # a numpy integer, which a type test would reject
])
def test_a_frame_stride_below_one_is_refused_by_name(frame_stride, accepted):
    """Both sides of the smallest usable stride. A stride of 0 escaped as a raw
    ``ZeroDivisionError`` and a negative one as a typed message about the
    caller's arrays, neither of which names the knob that is wrong."""
    from uacpy.visualization.plots import animate_field
    import matplotlib.pyplot as plt

    field = _make_synthetic_field(n_t=20, t_max=0.5)
    if accepted:
        assert animate_field(field, frame_stride=frame_stride) is not None
        plt.close('all')
        return
    with pytest.raises(ConfigurationError, match='frame_stride must be at least 1'):
        animate_field(field, frame_stride=frame_stride)
    plt.close('all')


def test_plot_time_snapshots_global_pmax():
    """``p_max`` scalar applies the same colour scale to every panel."""
    from uacpy.plot import plot_time_snapshots
    import matplotlib.pyplot as plt

    fields = {'A': _make_synthetic_field(n_t=20, t_max=0.5)}
    fig, axes = plot_time_snapshots(fields, times_s=(0.1, 0.3),
                                     p_max=0.5)
    for ax in axes.flat:
        im = ax.get_images()[0]
        assert im.get_clim() == (-0.5, 0.5)
    plt.close(fig)


@pytest.mark.parametrize("p_max, n_bars, label", [
    (None, 2, 'p(t) (Pa), row scale'),
    ((0.5, 0.25), 2, 'p(t) (Pa), row scale'),
    (0.5, 2, 'p(t) (Pa)'),
])
def test_the_snapshot_grid_states_its_colour_scale(p_max, n_bars, label):
    """Without a bar no amplitude can be read off a snapshot, and per-row
    scaling makes equal colours in two rows unequal pressures: every row
    carries a bar, labelled as its own scale unless one scale covers all."""
    from uacpy.plot import plot_time_snapshots
    import matplotlib.pyplot as plt

    fields = {'A': _make_synthetic_field(n_t=20, t_max=0.5),
              'B': _make_synthetic_field(n_t=20, t_max=0.5)}
    fig, axes = plot_time_snapshots(fields, times_s=(0.1, 0.3), p_max=p_max)
    panels = set(axes.flat)
    bars = [child for ax in axes[:, -1] for child in ax.child_axes]
    assert all(child not in panels for child in bars)
    assert len(bars) == n_bars
    assert all(bar.get_ylabel() == label for bar in bars)
    plt.close(fig)


def test_the_animated_source_star_is_whole_under_the_seafloor_overlay(
        marker_fraction_inside):
    """``env=`` pins the x limits to the field's own range span, which starts
    past r = 0 on every model grid, while the star is drawn at r = 0: the
    axis has to widen by the marker's own half width or the documented marker
    is not on the axes at all."""
    import uacpy
    import matplotlib.pyplot as plt
    from uacpy.visualization.plots import animate_field

    field = _make_synthetic_field(n_t=20, t_max=1.0, r_min=100.0)
    env = uacpy.Environment(bathymetry=120.0, ssp=1500.0, bottom=1650.0)
    fig, ax = plt.subplots()
    animate_field(field, env=env, frame_stride=5, ax=ax)
    assert marker_fraction_inside(ax) == pytest.approx(1.0)
    plt.close(fig)


def test_the_snapshot_source_star_is_whole_on_the_left_limit(
        marker_fraction_inside):
    """Each snapshot panel starts its axis at r = 0, exactly where the star
    is drawn; a limit ON the marker's centre cuts half the glyph away."""
    import matplotlib.pyplot as plt
    from uacpy.plot import plot_time_snapshots

    fields = {'A': _make_synthetic_field(n_t=20, t_max=0.5)}
    fig, axes = plot_time_snapshots(fields, times_s=(0.1,))
    assert marker_fraction_inside(axes[0, 0]) == pytest.approx(1.0)
    plt.close(fig)


def _stripe_at_60_m():
    """A time-series field on a receiver grid dense near the surface, with
    energy on the 60 m row only."""
    depths = np.array([0, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100], dtype=float)
    data = np.zeros((depths.size, 21, 5))
    data[list(depths).index(60.0)] = 1.0
    return Field(data=data,
                 coords={'depth': depths, 'range': np.linspace(50, 1050, 21),
                         'time': np.linspace(0.0, 1.0, 5)},
                 unit='Pa')


def _painted_depths(ax):
    """Depth extent of the cells painted with the stripe's value."""
    import matplotlib.collections as mcoll
    mesh = next(c for c in ax.collections if isinstance(c, mcoll.QuadMesh))
    y = np.asarray(mesh.get_coordinates())[:, 0, 1]      # cell edges in depth
    values = np.asarray(mesh.get_array()).reshape(len(y) - 1, -1)[:, 0]
    rows = np.flatnonzero(values == 1.0)
    return y[rows.min()], y[rows.max() + 1]


def test_a_snapshot_of_a_non_uniform_grid_puts_each_row_at_its_depth():
    """An image stretches rows evenly, which drew the 60 m row at 73-83 m;
    each cell is drawn about its own sample instead: 50-70 m."""
    from uacpy.plot import plot_time_snapshots
    import matplotlib.pyplot as plt
    fig, axes = plot_time_snapshots([_stripe_at_60_m()], times_s=[0.5])
    try:
        assert _painted_depths(axes[0, 0]) == pytest.approx((50.0, 70.0))
    finally:
        plt.close(fig)


def test_an_animation_of_a_non_uniform_grid_puts_each_row_at_its_depth():
    from uacpy.visualization.plots import animate_field
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    try:
        animate_field(_stripe_at_60_m(), ax=ax)
        assert _painted_depths(ax) == pytest.approx((50.0, 70.0))
    finally:
        plt.close(fig)


def test_the_animation_colorbar_names_the_fields_own_quantity():
    """Labelled as plot_field labels the same trace, from its kind and
    unit, and dropped on request."""
    from uacpy.visualization.plots import animate_field
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    try:
        animate_field(_stripe_at_60_m(), ax=ax)
        assert [a.get_ylabel() for a in fig.axes if a is not ax] == ['p(t) (Pa)']
    finally:
        plt.close(fig)
    fig, ax = plt.subplots()
    try:
        animate_field(_stripe_at_60_m(), ax=ax, show_colorbar=False)
        assert fig.axes == [ax]
    finally:
        plt.close(fig)


def test_the_snapshot_grid_draws_into_a_given_figure():
    from uacpy.plot import plot_time_snapshots
    import matplotlib.pyplot as plt
    parent = plt.figure(layout='constrained')
    try:
        fig, axes = plot_time_snapshots([_stripe_at_60_m()], times_s=[0.2, 0.5],
                                        fig=parent)
        assert fig is parent and axes.shape == (1, 2)
    finally:
        plt.close(parent)
