"""Plots for the ambient/biological/ship noise toolkit (uacpy.noise).

All noise plotting lives here; the ``uacpy.noise`` modules are pure computation
and import no matplotlib. Each function takes what it draws as its leading
positional arguments (a ``WenzNoise``, the frequency and weighting arrays
of a hearing group, the frequency and level arrays of a source spectrum, a
``RocCurve``). The target ``ax`` follows
them, so its position depends on how many there are — pass it by keyword; a
new figure is made when it is ``None``. Each returns ``(fig, ax)``, as
:func:`plot_field` does.
"""
import numpy as np
from uacpy.visualization.plots._common import (fig_ax, typed_plot_error,
                                               _carrier_or_arrays,
                                               _no_energy_as_gap,
                                               _require_nonempty, _title_or)
from uacpy.core.exceptions import ConfigurationError
from uacpy.visualization.plots.signal import _ref_label




@typed_plot_error
def plot_wenz(wenz, ax=None, *, show_components=True, title=None, ymin=6,
              ymax=146, figsize=(8, 5), **mpl_kw):
    """Plot a Wenz ambient-noise spectrum. Consumes a :class:`WenzNoise`
    result object (reads ``frequencies``/``total``/component arrays).
    ``**mpl_kw`` styles the total-noise line.

    Parameters
    ----------
    wenz : WenzNoise
        The spectrum to draw.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    show_components : bool, optional
        Draw each noise mechanism beside the total. Default True.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    ymin, ymax : float, optional
        Level-axis limits (dB). Default 6 and 146.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 5)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the total-noise line.
    """
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
                         f'{wenz.wind_speed_kn:.3g} kn, {wenz.rain_rate} rain)'))
    # setdefault, not hardcoded styles: **mpl_kw styles this line, and a caller
    # restyling it would otherwise collide and raise a raw TypeError.
    mpl_kw.setdefault('color', 'black')
    mpl_kw.setdefault('linewidth', 2.0)
    mpl_kw.setdefault('label', total_label)
    ax.semilogx(f, wenz.total, **mpl_kw)
    if show_components:
        components = (
            (wenz.shipping, 'blue',
             f'Shipping noise ({wenz.shipping_level} traffic)'),
            (wenz.wind, 'green', f'Wind noise ({wenz.wind_speed_kn:.3g} kn)'),
            (wenz.rain, 'orange', f'Rain noise ({wenz.rain_rate} rain)'),
            (wenz.thermal, 'red', 'Thermal noise'),
            (wenz.turbulence, 'purple', 'Turbulence noise'),
        )
        for level, colour, label in components:
            # A mechanism with no finite level (rain_rate='no') draws no
            # curve, so it takes no legend entry either.
            if np.any(np.isfinite(level)):
                ax.semilogx(f, level, color=colour, linestyle='dashed',
                            label=label)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Noise level (dB re 1 µPa²/Hz)')
    ax.set_title(_title_or(title, 'Wenz noise level estimate'))
    ax.set_xlim((f[0], f[-1]))
    ax.set_ylim((ymin, ymax))
    ax.legend()
    ax.grid(True)
    return fig, ax


@typed_plot_error
def plot_weighting(frequencies, weighting_dB=None, ax=None, *, label=None,
                   title=None, ymin=-40.0, ymax=5.0, figsize=(8, 4),
                   **mpl_kw):
    """Plot a marine-mammal auditory weighting curve ``W(f)`` (dB) on a log
    frequency axis, the values :func:`uacpy.noise.auditory_weighting`
    returns. Several groups share one axes through ``ax=``::

        f = np.logspace(1, 5.5, 400)
        fig, ax = plot_weighting(f, auditory_weighting(f, 'LF'), label='LF')
        plot_weighting(f, auditory_weighting(f, 'HF'), ax=ax, label='HF')

    Parameters
    ----------
    frequencies : array_like
        Frequencies (Hz).
    weighting_dB : array_like
        The weighting at each frequency (dB).
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    label : str, optional
        Legend label of the curve (the hearing group, say).
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    ymin, ymax : float, optional
        Weighting-axis limits (dB). Default -40 and 5; the high-frequency
        groups fall to -60 to -110 dB at the ends of 10 Hz-200 kHz, below
        the default floor, so widen it (``ymin=-120``) to see the whole
        curve.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 4)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the curve.
    """
    if isinstance(frequencies, str) or (
            isinstance(frequencies, (list, tuple)) and frequencies
            and all(isinstance(g, str) for g in frequencies)):
        raise ConfigurationError(
            "plot_weighting: draws a weighting curve it is handed, not a "
            "hearing group's name.",
            remediation="Compute it first: f = np.logspace(1, 5.5, 400); "
                        "plot_weighting(f, uacpy.noise.auditory_weighting("
                        "f, 'HF'), label='HF').")
    if weighting_dB is None:
        raise ConfigurationError(
            "plot_weighting: pass the frequencies and the weighting at each, "
            "plot_weighting(f, uacpy.noise.auditory_weighting(f, group)).")
    _require_nonempty('plot_weighting', frequencies=frequencies,
                      weighting_dB=weighting_dB)
    f = np.asarray(frequencies, dtype=float)
    w = np.asarray(weighting_dB, dtype=float)
    own_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    ax.semilogx(f, w, label=label, **mpl_kw)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Weighting W(f) (dB)")
    ax.set_title(_title_or(title, "Marine-mammal auditory weighting"))
    ax.set_ylim(ymin, ymax)
    ax.grid(which="both", alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    if own_fig:
        # Lay the plotter's own figure out around its labels, as
        # plot_constellation does: on the default figsize the x label
        # otherwise ends below the canvas.
        fig.tight_layout()
    return fig, ax


@typed_plot_error
def plot_source_level(frequencies, level_dB, ax=None, *, label=None, title=None,
                      figsize=(8, 4), **mpl_kw):
    """Plot a ship source-level spectrum (dB re 1 µPa²·m² vs band centre),
    the ISO 18405 unit :mod:`uacpy.noise` states its source levels in.

    Parameters
    ----------
    frequencies : array_like
        Band centre frequencies (Hz).
    level_dB : array_like
        Source level per band (dB re 1 µPa²·m²).
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    label : str, optional
        Legend label of the line.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 4)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the line.
    """
    _require_nonempty('plot_source_level', frequencies=frequencies, level_dB=level_dB)
    f = np.asarray(frequencies, dtype=float)
    lv = np.asarray(level_dB, dtype=float)
    own_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    # setdefault: a caller restyling the marker through **mpl_kw would
    # otherwise collide with the hardcoded one and raise a raw TypeError.
    mpl_kw.setdefault("marker", "o")
    ax.semilogx(f, lv, label=label, **mpl_kw)
    ax.set_xlabel("Decidecade band centre (Hz)")
    ax.set_ylabel("Source level (dB re 1 µPa²·m²)")
    ax.set_title(_title_or(title, "Ship radiated noise"))
    ax.grid(which="both", alpha=0.3)
    if label:
        ax.legend()
    if own_fig:
        # Lay the plotter's own figure out around its labels, as
        # plot_constellation does: on the default figsize the x label
        # otherwise ends below the canvas.
        fig.tight_layout()
    return fig, ax


@typed_plot_error
def plot_roc(pfa, pd=None, ax=None, *, label=None, title=None,
             figsize=(6.5, 5), **mpl_kw):
    """Receiver Operating Characteristic — detection probability ``P_D`` vs
    false-alarm probability ``P_F`` (log ``P_F`` axis).

    Draws a :class:`~uacpy.sonar.RocCurve` — ``plot_roc(curve)``, the call
    ``uacpy.sonar.roc_curve(d).plot()`` makes — or the two arrays. Several
    curves share one axes through ``ax=``::

        fig, ax = uacpy.sonar.roc_curve(1.0).plot()
        uacpy.sonar.roc_curve(3.0).plot(ax=ax)

    Parameters
    ----------
    pfa : RocCurve or array_like
        The curve, or the false-alarm probability of each point.
    pd : array_like, optional
        The detection probability of each point, with array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    label : str, optional
        Legend label of the line. ``None`` names a ``RocCurve`` by its
        deflection (``d'=2 (d=4.0)``, the detection index ``d = d'²``);
        array input has none.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(6.5, 5)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the line.
    """
    given = pfa
    _unpacked = _carrier_or_arrays(
        pfa, (pd,), count=2, who='plot_roc',
        carrier=(('RocCurve',), ('pfa', 'pd')), fields=('pfa', 'pd'))
    if _unpacked is not None:
        pfa, pd = _unpacked
        deflection = getattr(given, 'deflection', None)
        if label is None and deflection is not None:
            label = f"d'={deflection:.2g} (d={deflection ** 2:.1f})"
    elif pd is None:
        raise ConfigurationError(
            "plot_roc: pass one RocCurve, plot_roc(uacpy.sonar.roc_curve(d)), "
            "or both arrays, plot_roc(pfa, pd).")
    _require_nonempty('plot_roc', pfa=pfa, pd=pd)
    fig, ax = fig_ax(ax, figsize)
    ax.semilogx(np.asarray(pfa, float), np.asarray(pd, float), label=label,
                **mpl_kw)
    ax.set_xlabel("Probability of false alarm  $P_F$")
    ax.set_ylabel("Probability of detection  $P_D$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(_title_or(title, "Receiver operating characteristic"))
    ax.grid(which="both", alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize='small')
    return fig, ax


@typed_plot_error
def plot_cumulative_exposure(exposure, ax=None, *, criterion_dB=None,
                             criterion_label=None, label=None, title=None,
                             figsize=(9, 4), **mpl_kw):
    """The running sound exposure level of a record against time: the
    :class:`~uacpy.acoustics.CumulativeExposure`
    :func:`uacpy.acoustics.cumulative_sound_exposure_level` returns
    (``exposure.plot()`` draws the same picture).

    Parameters
    ----------
    exposure : CumulativeExposure
        One curve: a single record's, or one channel picked from a block.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    criterion_dB : float, optional
        An exposure criterion (dB re ``ref²``·s), drawn as a dashed rule:
        the time the curve crosses it is the time the criterion is met.
    criterion_label : str, optional
        Legend text of the rule; ``None`` is ``'criterion <value> dB'``.
    label : str, optional
        Legend label of the curve.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(9, 4)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the curve.
    """
    who = 'plot_cumulative_exposure'
    if type(exposure).__name__ != 'CumulativeExposure':
        raise ConfigurationError(
            f"{who}: expected a CumulativeExposure; got "
            f"{type(exposure).__name__}.",
            remediation="Compute it first: e = uacpy.acoustics."
                        "cumulative_sound_exposure_level(p, fs); then "
                        "e.plot().")
    levels = np.asarray(exposure.levels_dB, dtype=float)
    if levels.ndim != 1:
        raise ConfigurationError(
            f"{who}: this curve holds {levels.shape} values, a block of "
            f"records; one curve is drawn.",
            remediation="Accumulate one channel, or pick its row of "
                        "levels with the time axis kept.")
    own_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    ax.plot(exposure.times, _no_energy_as_gap(levels), label=label, **mpl_kw)
    if criterion_dB is not None:
        ax.axhline(float(criterion_dB), linestyle='--', color='k',
                   linewidth=0.8,
                   label=(f'criterion {float(criterion_dB):g} dB'
                          if criterion_label is None else criterion_label))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Cumulative SEL (dB re {_ref_label(exposure.ref)}Pa²·s)')
    ax.set_title(_title_or(title, 'Cumulative sound exposure level'))
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize='small')
    if own_fig:
        fig.tight_layout()
    return fig, ax
