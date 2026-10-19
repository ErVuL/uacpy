"""
Kraken Normal Mode Suite - one model, backend dispatcher

A single :class:`Kraken` wraps the AT Kraken pipeline, mirroring
``RAM``'s ``backend=`` convention:

- ``backend=`` selects the modes binary — ``'kraken'`` (real arithmetic)
  or ``'krakenc'`` (complex: elastic media / attenuation / leaky modes);
  ``None`` (default) picks krakenc when the environment carries shear,
  ``leaky_modes=True``, or a tabulated reflection coefficient (a
  ``'file'`` / ``'precalc'`` boundary or ``top_reflection_file=``), and
  kraken.exe otherwise.
- ``field.exe`` runs only when the requested run mode produces a field:
  ``compute_modes`` stops after the modes binary; ``compute_field`` /
  ``compute_transfer_function`` / ``compute_time_series`` chain field.exe.
- Range-dependence (bathy / SSP) is handled natively for field modes via
  ``field.exe`` adiabatic / coupled modes (``mode_coupling=``); the
  range-independent MODES path reduces the environment to one profile by
  the ``collapse=`` methods, else the package defaults (seabed median,
  bathymetry deepest, SSP and surface r = 0).

Note
----
The Acoustics Toolbox also carries KRAKEL (true elastic normal modes with
shear support using an FEM discretisation). Only its sources ship
(``third_party/Acoustics-Toolbox/Krakel/``): ``install.sh`` does not build
it (the AT Makefile leaves it out — it needs LAPACK), no ``krakel``
binary sits in ``uacpy/bin/oalib/``, and uacpy does not wrap it. Users
who need elastic modes can either:

* drive ``Kraken(backend='krakenc')`` (which handles elastic half-spaces
  via complex wavenumbers), or
* build ``krakel.exe`` from those sources and invoke it manually with a
  Kraken-format .env file.

Usage
-----
```python
from uacpy.models import Kraken, RunMode

kraken = Kraken()
modes = kraken.compute_modes(env, source)   # Modes (k, phi)
field = kraken.compute_field(env, source, receiver)       # Field (complex pressure)

# Complex modes for elastic bottom (auto, or force backend)
modes = Kraken(backend='krakenc').compute_modes(env_elastic, source)

# Range-dependent field via coupled modes
kraken = Kraken(mode_coupling='coupled', n_segments=10)
field = kraken.compute_field(env_rd, source, receiver)
```
"""

from uacpy.models.kraken._model import Kraken
from uacpy.models.kraken._settings import KrakenLaunch, KrakenSettings

__all__ = ['Kraken', 'KrakenLaunch', 'KrakenSettings']
