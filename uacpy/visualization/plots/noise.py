"""Plots for the ambient/biological/ship noise toolkit (uacpy.noise).

All noise plotting lives here; the ``uacpy.noise`` modules are pure computation
and import no matplotlib. Each function takes the target ``ax`` as its second
positional argument (a new figure is made when it is ``None``) and returns
``(fig, ax)`` — the same convention as :func:`plot_field`.
"""
import numpy as np
from uacpy.visualization.plots._common import fig_ax, typed_plot_error, _require_nonempty
from uacpy.core.exceptions import ConfigurationError




@typed_plot_error
def plot_wenz(wenz, ax=None, *, show_components=True, title=None, ymin=6,
              ymax=146, figsize=(8, 5), **mpl_kw):
    """Plot a Wenz ambient-noise spectrum. Consumes a :class:`WenzNoise`
    result object (reads ``frequencies``/``total``/component arrays).
    ``**mpl_kw`` styles the total-noise line."""
    if not hasattr(wenz, 'frequencies') or not hasattr(wenz, 'total'):
        raise ConfigurationError(
            f"plot_wenz: expected a WenzNoise result (with .frequencies and "
            f".total); got {type(wenz).__name__}.")
    f = wenz.frequencies
    fig, ax = fig_ax(ax, figsize)
    # The total-noise line names every condition it sums when it is the only
    # curve on the panel; alongside the components it names just the depth
    # regime, which is the one condition no component line reports.
    total_label = (f'Total noise ({wenz.water_depth} water)' if show_components
                   else (f'Total noise ({wenz.water_depth} water, '
                         f'{wenz.shipping_level} traffic, '
                         f'{wenz.wind_speed_kn:g} kn, {wenz.rain_rate} rain)'))
    # setdefault, not hardcoded styles: **mpl_kw styles this line, and a caller
    # restyling it would otherwise collide and raise a raw TypeError.
    mpl_kw.setdefault('color', 'black')
    mpl_kw.setdefault('linewidth', 2.0)
    mpl_kw.setdefault('label', total_label)
    ax.semilogx(f, wenz.total, **mpl_kw)
    if show_components:
        ax.semilogx(f, wenz.shipping, color='blue', linestyle='dashed',
                    label=f'Shipping noise ({wenz.shipping_level} traffic)')
        ax.semilogx(f, wenz.wind, color='green', linestyle='dashed',
                    label=f'Wind noise ({wenz.wind_speed_kn:g} kn)')
        ax.semilogx(f, wenz.rain, color='orange', linestyle='dashed',
                    label=f'Rain noise ({wenz.rain_rate} rain)')
        ax.semilogx(f, wenz.thermal, color='red', linestyle='dashed',
                    label='Thermal noise')
        ax.semilogx(f, wenz.turbulence, color='purple', linestyle='dashed',
                    label='Turbulence noise')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'Noise Level (dB re 1$\mu$Pa$^2$/Hz)')
    ax.set_title(title or 'WENZ noise level estimate', loc='left')
    ax.set_xlim((f[0], f[-1]))
    ax.set_ylim((ymin, ymax))
    ax.legend()
    ax.grid(True)
    return fig, ax


@typed_plot_error
def plot_weighting(group, ax=None, *, frequency=None, title=None,
                   figsize=(8, 4), **mpl_kw):
    """Plot marine-mammal auditory weighting curve(s). ``group`` is a name or a
    list of names."""
    from uacpy.noise.marine_mammal import auditory_weighting
    groups = [group] if isinstance(group, str) else list(group)
    f = (np.logspace(1, 5.5, 400) if frequency is None
         else np.asarray(frequency, dtype=float))
    fig, ax = fig_ax(ax, figsize)
    for g in groups:
        ax.semilogx(f, auditory_weighting(f, g), label=f"{g.upper()}", **mpl_kw)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Weighting W(f) (dB)")
    ax.set_title(title or "Marine-mammal auditory weighting", loc="left")
    ax.set_ylim(-40, 5)
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    return fig, ax


@typed_plot_error
def plot_source_level(frequency, level_db, ax=None, *, label=None, title=None,
                      figsize=(8, 4), **mpl_kw):
    """Plot a ship source-level spectrum (dB re 1 µPa·m vs band centre)."""
    _require_nonempty('plot_source_level', frequency=frequency, level_db=level_db)
    f = np.asarray(frequency, dtype=float)
    lv = np.asarray(level_db, dtype=float)
    fig, ax = fig_ax(ax, figsize)
    # setdefault: a caller restyling the marker through **mpl_kw would
    # otherwise collide with the hardcoded one and raise a raw TypeError.
    mpl_kw.setdefault("marker", "o")
    ax.semilogx(f, lv, label=label, **mpl_kw)
    ax.set_xlabel("Decidecade band centre (Hz)")
    ax.set_ylabel("Source level (dB re 1 µPa·m)")
    ax.set_title(title or "Ship radiated noise", loc="left")
    ax.grid(which="both", alpha=0.3)
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_roc(deflection=None, ax=None, *, pfa=None, pd=None, n_points=200,
             label=None, title=None, figsize=(6.5, 5), **mpl_kw):
    """Receiver Operating Characteristic — detection probability ``P_D`` vs
    false-alarm probability ``P_F`` (log ``P_F`` axis).

    Provide either a detector **deflection** ``d'`` — a scalar or sequence, one
    ROC curve per value computed via :func:`uacpy.sonar.roc_curve` (the detection
    index is ``d = d'^2``) — or pre-computed ``pfa``/``pd`` arrays. Consumes the
    ``uacpy.sonar`` detection theory; returns ``(fig, ax)`` like the other
    plotters."""
    fig, ax = fig_ax(ax, figsize)
    if pfa is not None and pd is not None:
        ax.semilogx(np.asarray(pfa, float), np.asarray(pd, float),
                    label=label, **mpl_kw)
    elif deflection is not None:
        from uacpy.sonar import roc_curve
        defl = np.atleast_1d(np.asarray(deflection, dtype=float))
        for d in defl:
            pf, pdc = roc_curve(float(d), n_points=n_points)
            lab = (label if (label and defl.size == 1)
                   else f"d'={float(d):.2g} (d={float(d) ** 2:.1f})")
            ax.semilogx(pf, pdc, label=lab, **mpl_kw)
    else:
        raise ConfigurationError(
            "plot_roc: pass deflection= (d', scalar or sequence) or both "
            "pfa= and pd= arrays.")
    ax.set_xlabel("Probability of false alarm  $P_F$")
    ax.set_ylabel("Probability of detection  $P_D$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title or "Receiver operating characteristic", loc="left")
    ax.grid(which="both", alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=8)
    return fig, ax
