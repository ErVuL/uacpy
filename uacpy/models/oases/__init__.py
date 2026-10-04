"""
OASES - Ocean Acoustics and Seismic Exploration Synthesis

OASES is a comprehensive acoustic/seismic propagation modeling suite developed
by Henrik Schmidt at MIT. It includes multiple executables for different scenarios:

- **OAST**: Transmission Loss via wavenumber integration
- **OASN**: Noise covariance matrices and signal replicas (matched-field processing)
- **OASR**: Reflection coefficients at stratified interfaces
- **OASP**: Broadband transfer-function / pulse synthesis (wideband wavenumber integration)
- **OASS**: Reverberation / scattered-field statistics from rough interfaces
- **OASSP**: Broadband realizations of the scattered field (OASS's time-domain counterpart)

This module provides Python wrappers for all OASES executables following
the UACPY propagation model architecture.

Usage
-----
```python
from uacpy.models import OAST, OASN, OASR, OASP, OASS, OASSP

# Transmission loss using OAST
oast = OAST()
result = oast.run(env, source, receiver)

# Noise covariance / replicas using OASN
oasn = OASN()
cov = oasn.run(env, source, receiver, run_mode=RunMode.COVARIANCE)

# Reflection coefficients using OASR
oasr = OASR()
refl = oasr.run(env, source, receiver, run_mode=RunMode.REFLECTION)

# Broadband pulse synthesis using OASP (wavenumber integration)
oasp = OASP()
result = oasp.run(env, source, receiver, run_mode=RunMode.BROADBAND)

# Reverberation from a rough interface using OASS (runs its mean field first)
oass = OASS(correlation_length=10.0)
reverb = oass.run(env, source, receiver)

# One realization of the scattered field using OASSP (runs OASP first)
oassp = OASSP(correlation_length=5.0)
h_scattered = oassp.run(env, source, receiver, run_mode=RunMode.BROADBAND)
```
"""

from uacpy.models.oases._base import OASES
from uacpy.models.oases.oast import OAST, OASTSettings
from uacpy.models.oases.oasn import OASN, OASNSettings
from uacpy.models.oases.oasr import OASR, OASRSettings
from uacpy.models.oases.oasp import OASP, OASPSettings
from uacpy.models.oases.oass import OASS, OASSSettings
from uacpy.models.oases.oassp import OASSP, OASSPSettings

__all__ = [
    'OASES',
    'OAST',
    'OASN',
    'OASR',
    'OASP',
    'OASS',
    'OASSP',
    'OASTSettings',
    'OASNSettings',
    'OASRSettings',
    'OASPSettings',
    'OASSSettings',
    'OASSPSettings',
]
