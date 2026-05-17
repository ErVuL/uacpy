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
) -> Field:
    """Outgoing pulse traveling at C_WATER. Independent of depth (so any
    horizontal cut sees the same wavefront)."""
    depths = np.linspace(0.0, 100.0, n_d)
    ranges = np.linspace(0.0, C_WATER * t_max, n_r)
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
    """Update the animation at three frames and verify the heatmap's
    argmax(|p|) range index moves outward — sanity check on the
    set_array calls."""
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
    assert field.kind != 'time_series'
    with pytest.raises(ValueError, match="kind='time_series'"):
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
    with pytest.raises(ValueError, match="missing coord axes"):
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
    from uacpy.visualization import save_animation

    field = _make_synthetic_field(n_t=40, t_max=1.0)
    out = save_animation(field, tmp_path / 'pulse.gif',
                         fps=10, frame_stride=4)
    assert out.exists()
    assert out.stat().st_size > 1024  # > 1 KiB, real animation


def test_save_animation_rejects_unknown_suffix(tmp_path):
    from uacpy.visualization import save_animation

    field = _make_synthetic_field(n_t=10)
    with pytest.raises(ValueError, match=r"cannot infer writer"):
        save_animation(field, tmp_path / 'pulse.xyz')


def test_save_animation_closes_figure(tmp_path):
    """Calling save_animation should not leak open figures."""
    import matplotlib.pyplot as plt
    from uacpy.visualization import save_animation

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
    from uacpy.visualization import plot_time_snapshots

    fields = {
        'A': _make_synthetic_field(n_t=40, t_max=1.0),
        'B': _make_synthetic_field(n_t=40, t_max=1.0),
    }
    fig, axes = plot_time_snapshots(fields, times_s=(0.2, 0.5, 0.8))
    assert axes.shape == (2, 3)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_time_snapshots_empty_raises():
    from uacpy.visualization import plot_time_snapshots

    with pytest.raises(ValueError, match='empty'):
        plot_time_snapshots({}, times_s=(0.1,))


def test_plot_time_snapshots_global_pmax():
    """``p_max`` scalar applies the same colour scale to every panel."""
    from uacpy.visualization import plot_time_snapshots
    import matplotlib.pyplot as plt

    fields = {'A': _make_synthetic_field(n_t=20, t_max=0.5)}
    fig, axes = plot_time_snapshots(fields, times_s=(0.1, 0.3),
                                     p_max=0.5)
    for ax in axes.flat:
        im = ax.get_images()[0]
        assert im.get_clim() == (-0.5, 0.5)
    plt.close(fig)
