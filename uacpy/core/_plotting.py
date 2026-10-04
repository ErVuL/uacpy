"""The one import that reaches from the layers below :mod:`uacpy.visualization`
up into it.

Every ``.plot()`` (and ``plot_*``) method of a carrier or a result draws
through a public plotter of :mod:`uacpy.visualization`, looked up here by
name. ``uacpy.visualization`` imports ``uacpy.core`` at module scope to render
the core types, so the import runs when a figure is drawn: at file scope it
would make ``import uacpy`` raise ImportError from a partially initialised
module (docs/DEV.md section 7).
"""


def plotter(name: str):
    """The public plotter ``uacpy.plot.<name>``, read from its
    implementation module ``uacpy.visualization.plots``."""
    # Deferred into the body: ``uacpy.visualization`` imports ``uacpy.core``
    # at module scope, so this line at file scope makes ``import uacpy``
    # raise ImportError.
    from uacpy.visualization import plots
    return getattr(plots, name)
