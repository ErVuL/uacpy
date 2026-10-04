"""
RAM - Range-dependent Acoustic Model wrapper (multi-backend dispatcher)

The :class:`RAM` class auto-selects one of four vendored Collins-family PE
binaries based on the environment:

- **mpiramS** (default — fluid bottom + flat surface): Dushaw's Fortran 90/95
  rewrite of Collins' original RAM. Native broadband Q/T loop, MPI-ready
  upstream (uacpy builds the serial variant). Custom `inpe`/SSP/BTH multi-file
  input format via :mod:`uacpy.io.mpirams_writer`.
- **rams0.5** (any ``shear_speed > 0`` anywhere): Collins' RAMS elastic PE
  for sediments with shear waves. Collins-style ``rams.in`` input via
  :mod:`uacpy.io.ramsurf_writer`.
- **ramsurf1.5** (``env.altimetry is not None``): Collins' rough-surface /
  beach-geometry PE. Same writer as rams.
- **ramgeo** (fluid + flat surface with a layered bottom, narrowband):
  Collins' RAMGeo range-dependent-geoacoustics PE — its sediment layers
  parallel the bathymetry rather than lying flat (``ramgeo1.5.f:3-4``).

Elastic bottom + altimetry raises ``UnsupportedFeatureError`` — no published
Collins PE handles that combination; use OASES for range-independent elastic.

Run modes by backend:
- mpiramS: ``COHERENT_TL``, ``BROADBAND``, ``TIME_SERIES`` (native Q/T loop).
- rams0.5 / ramsurf1.5 / ramgeo: ``COHERENT_TL`` natively, plus ``BROADBAND``
  and ``TIME_SERIES`` via the patched complex-envelope output
  (``pcomplex.bin``), one binary launch per frequency bin
  (:func:`~uacpy.models.ram.collins.resolve_collins_band` resolves the grids,
  one launch each: :meth:`RAM._n_launches`). See
  ``third_party/MODIFICATIONS.md`` for the upstream patch.

Every run goes through the stages of :class:`~uacpy.models.base.PropagationModel`:
the backend and each launch's PE grid are resolved once, before anything is
written (``RAM().run_settings(env, source, receiver).engine``, a
:class:`RamSettings`), and the decks are written from that record.

The lower boundary at zmax is an absorbing layer in all four backends, not a
rigid Neumann floor: mpiramS ramps the sediment attenuation to
``absorber_attenuation`` over the deepest ``absorber_width_wavelengths``
wavelengths of the PE domain, and ``ram.collins.collins_range_segments`` writes
the same ramp into each Collins profile section.
"""

from uacpy.models.ram._model import RAM
from uacpy.models.ram._settings import RamGrid, RamSettings

__all__ = ['RAM', 'RamGrid', 'RamSettings']
