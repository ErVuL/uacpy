"""Which RAM backend runs a call: the preference among the Collins
binaries, a forced backend's refusals, and the knobs one backend ignores."""

from uacpy.core.run_settings import RunMode
from uacpy.core.environment import (
    Environment,
)
from uacpy.core.exceptions import ConfigurationError


def prefer_ramgeo(env: Environment, run_mode) -> bool:
    """RAMGEO is auto-selected for a narrowband (COHERENT_TL) fluid,
    flat-surface environment whose bottom is layered. A simple half-space
    is still accepted when ``backend='ramgeo'`` is forced.

    What makes it the better layered backend is how the deck REPRESENTS the
    stack, not whether the stack follows the seabed — mpiramS anchors its
    own sediment profile at the local seafloor too (``zwork(2) = depth``,
    ``mpiramS/src/ram.f90:334-342``), so "layers track the bathymetry" is
    true of both and separates nothing. Two things do separate them:

    * **Native per-layer breakpoints.** ramgeo's ``zread``
      (``ramgeo1.5.f:209-235``) assigns each ``(depth, value)`` pair of the
      block to node ``1.5 + z/dz`` and fills linearly only *between* them,
      and uacpy emits one pair per layer top and base — so a layer is a
      constant run and each interface is a one-cell step. mpiramS instead
      samples the column onto a UNIFORM ladder of ``nzs`` control points
      (``mpirams.sample_layered_column``) and ``gorp`` (``ram.f90:373-403``)
      interpolates between those, so every interface is a ramp one
      control-point interval wide, wherever the interface happens to fall.
    * **The interval that ramp is wide.** That ladder spans ``sedlayer``,
      which grows with the domain depth; with a fixed ``nzs`` the
      interface resolution degrades as ``zmax`` grows (1.04 dB rms at
      zmax=700 m against 400 m). :func:`mpirams.prepare_bottom_properties`
      sizes ``nzs`` from ``sedlayer/dz`` to hold that interval at one
      depth cell, which bounds the effect but does not remove the ramp.
    """
    if run_mode is not None and run_mode != RunMode.COHERENT_TL:
        return False
    return env.bottom.is_layered


def validate_forced_backend(backend: str, env: Environment) -> None:
    """Reject a forced ``backend=`` that cannot represent ``env``."""
    elastic = env.bottom.is_elastic
    rough = env.altimetry is not None
    if backend in ('mpirams', 'ramgeo', 'ramsurf') and elastic:
        raise ConfigurationError(
            f"RAM(backend={backend!r}) is a fluid PE and cannot model the "
            f"elastic bottom (shear>0) in this environment. Use "
            f"backend='rams', or backend=None for automatic dispatch."
        )
    if backend in ('mpirams', 'ramgeo', 'rams') and rough:
        raise ConfigurationError(
            f"RAM(backend={backend!r}) models a flat pressure-release "
            f"surface and cannot honour env.altimetry. Use "
            f"backend='ramsurf', or backend=None for automatic dispatch."
        )
    if backend == 'ramsurf' and not rough:
        raise ConfigurationError(
            "RAM(backend='ramsurf') needs a variable surface "
            "(env.altimetry). For a flat surface use backend='mpirams' / "
            "'ramgeo', or backend=None for automatic dispatch."
        )
    if backend == 'rams' and not elastic:
        # rams0.5 is the elastic (RAMS) PE. Run on a fluid bottom its
        # shear machinery degenerates and it returns a null field —
        # TL saturated at 200 dB at every range — rather than failing.
        # Mirrors the ramsurf rule above: a backend whose defining
        # feature is absent from the env is a configuration error.
        raise ConfigurationError(
            "RAM(backend='rams') is the elastic PE and needs a bottom "
            "with shear (shear_speed > 0); on a fluid bottom it returns "
            "a null field. Use backend='mpirams' / 'ramgeo', or "
            "backend=None for automatic dispatch."
        )
    if backend == 'rams':
        check_rams_top_layer_carries_shear(env)


def check_rams_top_layer_carries_shear(env: Environment) -> None:
    """Refuse a rams march whose seabed starts with a zero-shear layer.

    ``rams0.5.f:593-600`` builds the fluid-solid interface rows by dividing
    by ``mub(iz+1)``, the shear modulus of the first sediment node below
    the seafloor (``mub = rhob·cs²`` at ``:204``). A top layer left at the
    ``SedimentLayer`` default ``shear_speed=0`` over an elastic
    half-space — "sediment over rock" — therefore writes ``cs=0`` at that
    node and the march divides by zero: every sample comes back NaN and
    no ``dz``/``n_pade``/``theta`` choice changes it. Deeper zero-shear
    layers are fine: the solid-layer rows (``:491-516``) only difference
    ``mub`` and never divide by it.

    A column with no layers is its half-space from the seafloor down, so
    a fluid half-space column beside an elastic one — sand next to a rock
    outcrop in a range-dependent ``Bottom`` — writes ``cs=0`` at the same
    node from the section where that column starts, and every sample
    from there on is NaN.
    """
    multi = len(env.bottom.columns) > 1
    for i, col in enumerate(env.bottom.columns):
        if (not col.layers
                and float(col.halfspace.shear_speed or 0.0) <= 0.0):
            where = f"column {i}" if multi else "the seabed"
            raise ConfigurationError(
                f"RAM:rams cannot march this seabed: {where} has no "
                f"layers and a fluid half-space (shear_speed=0) where "
                f"another column is elastic. rams0.5.f:593-600 divides by "
                f"the shear modulus of the first sediment node below the "
                f"seafloor, so every sample from the section that column "
                f"starts is NaN; no grid or Padé setting changes that.",
                remediation=("Give every column's surficial material a "
                             "shear speed (shear_speed > 0) so rams "
                             "marches the whole track as a solid, or set "
                             "every shear_speed to 0 so a fluid PE "
                             "(mpiramS / ramgeo) runs instead."),
            )
        if col.layers and float(col.layers[0].shear_speed) <= 0.0:
            where = (f"column {i} " if multi else "")
            raise ConfigurationError(
                f"RAM:rams cannot march this seabed: {where}layer 0 "
                f"(thickness {float(col.layers[0].thickness):g} m) has "
                f"shear_speed=0 over an elastic stack. rams0.5.f:593-600 "
                f"divides by the shear modulus of the first sediment node "
                f"below the seafloor, so a fluid top layer makes every "
                f"sample NaN; no grid or Padé setting changes that.",
                remediation=("Give the top layer a shear speed "
                             "(shear_speed > 0) so rams marches it as a "
                             "solid, or set every shear_speed to 0 so the "
                             "layered fluid PE (ramgeo) runs instead."),
            )


def backend_origin(env: Environment, backend: str, *, knobs) -> str:
    """Why :meth:`select_backend` chose ``backend``, for
    :attr:`RamSettings.backend_origin`."""
    if knobs.backend is not None:
        return 'RAM(backend=…)'
    if backend == 'rams':
        return 'select_backend: an elastic seabed'
    if backend == 'ramsurf':
        return 'select_backend: env.altimetry'
    if backend == 'ramgeo':
        return 'select_backend: a layered fluid seabed, COHERENT_TL'
    return 'select_backend: a fluid seabed under a flat surface'
