"""The reduction vocabularies of the range-dependent carriers.

The methods that collapse a range axis, a bathymetry and one column's
sediment-layer stack, shared by the carriers that implement them and the
model layer's collapse policy that names them.
"""

__all__ = [
    'RANGE_COLLAPSE_METHODS', 'DEPTH_COLLAPSE_METHODS',
    'COLUMN_COLLAPSE_METHODS',
]


#: Methods that reduce a range-dependent carrier to a single column.
#: One vocabulary, three implementations —
#: :meth:`uacpy.SoundSpeedProfile.collapse_range`,
#: :meth:`uacpy.Surface.collapse_range` and :meth:`uacpy.Bottom.collapse_range` —
#: plus the model layer's ``VALID_COLLAPSE_METHODS`` entries for ``'ssp'``,
#: ``'bottom_range'`` and ``'surface'``, which are built from this name. It
#: lives here rather than in any one carrier because none of the three owns
#: the concept and all three already import this module: spelled out per
#: carrier, a method added to the model layer's registry and not to the
#: carrier that implements it is accepted at construction and raises from
#: inside a writer at ``run()``-time, which is the failure the registry
#: exists to prevent.
RANGE_COLLAPSE_METHODS = ('r0', 'rmax', 'mean', 'median')


#: Methods that reduce a range-dependent bathymetry to one depth, shared by
#: :meth:`uacpy.Bathymetry.collapse_range` and the model layer's
#: ``'bathymetry'`` entry. Ordered loudest-first because the default is
#: ``'max'``: a model that cannot take a sloping bottom is safer run against
#: the deepest water than the shallowest.
DEPTH_COLLAPSE_METHODS = ('max', 'median', 'mean', 'min', 'initial')


#: Methods that flatten one column's sediment-layer stack to a half-space,
#: shared by :meth:`uacpy.core.bottom.SeabedColumn.collapse_layers` and the model
#: layer's ``'bottom_layers'`` entry. See that method for what each keeps.
COLUMN_COLLAPSE_METHODS = ('halfspace', 'top_layer', 'volume_average')


def _method_list(methods) -> str:
    """The vocabulary as an error message spells it: ``'r0', 'rmax', ...``.

    Built from the constant so a message cannot name a method the carrier
    stopped accepting, which is how these vocabularies drifted before.
    """
    return ', '.join(repr(m) for m in methods)
