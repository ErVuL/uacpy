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

import re
import warnings

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union

from uacpy.models.base import (
    PropagationModel, RunMode, ModelSpec, USER_FRAME_SKIP,
    VALID_SOURCE_TYPES,
)
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import (
    Result, Field,
    Covariance, Replicas, ReflectionCoefficient,
)
from uacpy.core.exceptions import (
    ConfigurationError,
    UnsupportedFeatureError,
)
from uacpy.io.oases_writer import (
    write_oast_input, write_oasn_input, write_oasp_input, write_oasr_input,
    write_oass_input, write_oassp_input, oases_wavenumber_bounds,
    oass_bottom_interfaces, bottom_interface_roughness,
)
from uacpy.io.units import m_to_km
from uacpy.io.oases_reader import (
    read_oast_tl,
    read_oasn_covariance,
    read_oasn_replicas,
    read_oases_rhs_header,
    read_oasp_trf,
    read_oasr_reflection_coefficients,
)


def _oases_resample_frequencies(
    freqs: np.ndarray, model_name: str, log_spaced: bool = False,
) -> tuple[float, float, int, bool]:
    """Convert an arbitrary user ``frequencies=`` vector to the
    ``(fmin, fmax, N)`` triple OASR/OASP write into the input file.

    OASR's ``.dat`` and OASP's broadband kernel both express their
    frequency axis as a min/max/count, which implies equispaced
    sampling. If the user passes a non-equispaced vector we warn and
    auto-resample onto ``np.linspace(fmin, fmax, N)`` so the result's
    ``Result.frequencies`` reflects what the model actually saw.

    Returns
    -------
    fmin, fmax : float
    n : int
        Number of equispaced bins.
    resampled : bool
        True iff the user vector was non-equispaced and got resampled.

    Raises
    ------
    ConfigurationError
        When ``log_spaced`` is set and the vector's lowest frequency is not
        strictly positive: the OASR kernel takes ``LOG(FREQ1)``.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    fmin = float(freqs.min())
    fmax = float(freqs.max())
    n = int(freqs.size)
    resampled = False
    if n > 1 and log_spaced:
        # OASR option 'C' makes the kernel sweep LOGARITHMIC
        # (unoasr21.f:123-125, :243) — it is not a plot option. The
        # equispacing test below is then exactly inverted: a linear request
        # is the one being silently regridded, and a log request is correct.
        # A log sweep has no grid to sit on unless every frequency is
        # positive: the kernel takes F1LOG = LOG(FREQ1) (unoasr21.f:124) and
        # the binary aborts a zero bound one line earlier
        # (:116-118, "FREQUENCIES MUST BE NON-ZERO") while a negative one
        # reaches LOG unchecked. Refused here so the deck is never written.
        # NaN-closed: nan fails `> 0` both ways.
        if not (fmin > 0):
            raise ConfigurationError(
                f"{model_name}: option 'C' makes the frequency sweep "
                f"logarithmic (unoasr21.f:123-125), which needs every "
                f"frequency strictly positive; the frequencies= vector "
                f"starts at {fmin}. Raise the lower bound above 0 Hz, or "
                f"drop 'C' for a linear sweep."
            )
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = freqs[1:] / freqs[:-1]
        target = (fmax / fmin) ** (1.0 / (n - 1))
        if not np.allclose(ratios, target, rtol=1e-6):
            resampled = True
            grid = np.geomspace(fmin, fmax, n)
            warnings.warn(
                f"{model_name}: option 'C' makes the frequency sweep "
                f"logarithmic (unoasr21.f:123-125 computes the coefficients at "
                f"exp-spaced frequencies), but the frequencies= vector is not "
                f"log-spaced. It will be evaluated at "
                f"np.geomspace({fmin}, {fmax}, {n}) — "
                f"{np.array2string(grid, precision=4, max_line_width=200)} — "
                f"not at the values given. Pass a log-spaced vector, or drop "
                f"'C' for a linear sweep.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        return fmin, fmax, n, resampled
    if n > 1:
        diffs = np.diff(freqs)
        # Equispaced if all diffs match the mean diff within tolerance
        # (rtol scaled to the band; atol is a tiny absolute floor).
        target = (fmax - fmin) / (n - 1)
        if not np.allclose(diffs, target, rtol=1e-6, atol=1e-9):
            resampled = True
            warnings.warn(
                f"{model_name}: frequencies= vector is non-equispaced; "
                f"OASES expresses the frequency axis as (fmin, fmax, N) "
                f"so the vector has been resampled onto "
                f"np.linspace({fmin}, {fmax}, {n}). "
                f"Result.frequencies will be the resampled grid, not "
                f"your input. Pass an equispaced vector (or "
                f"freq_min/freq_max/n_frequencies) to suppress.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
    return fmin, fmax, n, resampled


def _warn_if_trf_grid_replaced_request(requested, produced) -> None:
    """Warn when OASP's ``.trf`` grid is not the frequency the caller asked.

    OASP is a *pulse* model: its frequency axis is the FFT ladder implied by
    the time window and sample rate, not the bins the caller listed. The
    ``(fmin, fmax, N)`` triple in the deck bounds the computation, but the
    ``.trf`` comes back on the ladder — measured, ``frequencies=[150, 200,
    250]`` returns 820 bins spanning 149.902-249.878 Hz. ``Kraken`` on the
    identical call returns exactly the three requested, so a caller who has
    used one has no reason to expect the other.

    Not an error: the ladder is what OASP computes and it is usable. But the
    caller asked for specific bins and silently got a different axis, so the
    substitution has to be visible. Both OASP branches call this: BROADBAND
    with the whole ladder, COHERENT_TL with the single nearest bin its Field
    is stamped with, hence the two phrasings below. ``Field.at(frequency=…)``
    picks the nearest ladder bin.
    """
    if requested is None:
        return
    req = np.atleast_1d(np.asarray(requested, dtype=float))
    got = np.atleast_1d(np.asarray(produced, dtype=float))
    if req.size == got.size and np.allclose(req, got, rtol=1e-6, atol=1e-9):
        return
    if req.size == 1 and got.size == 1:
        warnings.warn(
            f"OASP: the run asked for {req[0]:g} Hz, but OASP's internal FFT "
            f"ladder has no bin there; the Field carries the nearest bin, "
            f"{got[0]:.5f} Hz. OASP is a pulse model — its frequency axis "
            f"follows the time window — and the phase error of the "
            f"substituted bin grows with range. Use Kraken/Scooter for "
            f"exactly the frequency named, or pick n_time_samples/freq_max "
            f"so a ladder bin lands on it.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        return
    warnings.warn(
        f"OASP: frequencies= asked for {req.size} bin(s) spanning "
        f"{req.min():g}-{req.max():g} Hz; the .trf returns OASP's own FFT "
        f"ladder of {got.size} bin(s) spanning {got.min():g}-{got.max():g} Hz, "
        f"which is the axis on the Field. OASP is a pulse model — its "
        f"frequency axis follows the time window, not the requested list. Use "
        f"Field.at(frequency=…) to take the nearest bin, or Kraken/Scooter if "
        f"you need exactly the frequencies you name.",
        UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def _mask_zero_range(field, source, model_name: str):
    """NaN the ``r <= 0`` column(s) of a point-source OASES field.

    OASES evaluates its wavenumber integral under the asymptotic Hankel
    carrier, whose ``1/sqrt(r)`` cylindrical spreading is singular on the
    source axis, so the number it returns at ``r = 0`` belongs to no range
    (measured on a 100-m Pekeris case: OASP |p| = 1.23 at r = 0 against
    5e-3 at 500 m; OASS 56.3 dB at r = 0). Kraken (``_mask_zero_range``),
    Scooter and RAM mask the same cells, which keeps the family's grids
    comparable cell by cell. Line/scaled sources carry no ``1/sqrt(r)``
    and are left alone, as they are in the sibling models.

    The line/scaled branch is unreachable through any OASES ``run()``: all six
    classes declare ``source_types = ('point',)`` and ``validate_inputs``
    raises ``UnsupportedFeatureError`` on anything else before every call
    site. It is kept as the behaviour a widened ``spec.source_types`` would
    need — without it, an OASES line source would have its ``r = 0`` column
    NaN'd for a spreading factor it does not carry. ``test_oases.py`` drives
    it directly.
    """
    if getattr(source, 'source_type', 'point') != 'point':
        return field
    ranges = np.atleast_1d(np.asarray(field.coords['range'], dtype=float))
    zero = ranges <= 0.0
    if not zero.any():
        return field
    warnings.warn(
        f"{model_name}: {int(zero.sum())} receiver range(s) at r <= 0, "
        "where the point-source cylindrical-spreading factor 1/sqrt(r) is "
        "singular; those columns are returned as NaN (no data). Move the "
        "receiver off the source axis (e.g. r = 1 m) to get a field value.",
        UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )
    data = np.array(field.data)
    if not np.issubdtype(data.dtype, np.inexact):
        data = data.astype(float)
    data[:, zero, ...] = np.nan
    field.data = data
    return field


def _stack_oasr_data(data: dict):
    """Convert per-frequency lists from :func:`read_oasr_reflection_coefficients`
    into typed-Result arrays.

    Returns
    -------
    theta : ndarray, shape ``(n_angles,)``
    R : ndarray, shape ``(n_angles,)`` (single freq) or ``(n_angles, n_frequencies)``
    phi : ndarray, same shape as ``R``  — phase in radians (reader stores degrees)
    freqs : ndarray, shape ``(n_frequencies,)``
    """
    freqs = np.asarray(data.get('frequencies', []), dtype=float)
    angle_lists = data.get('angles_or_slowness', [])
    R_lists = data.get('magnitude', [])
    phi_lists = data.get('phase', [])
    if not angle_lists:
        raise ConfigurationError(
            f"OASR reader returned no frequency samples: the file carries "
            f"{freqs.size} frequency value(s) and no angle/slowness rows. "
            f"Re-run OASR with 'T' in the option string — the reflection "
            f"coefficient table is written only under that letter.")
    theta = np.asarray(angle_lists[0], dtype=float)
    n_angles = len(theta)
    n_frequencies = len(freqs)
    for i, (a, m, p) in enumerate(zip(angle_lists, R_lists, phi_lists)):
        if len(a) != n_angles:
            raise ConfigurationError(
                f"OASR: angle grid mismatch between frequencies "
                f"(freq[{i}] has {len(a)} angles, freq[0] has {n_angles}). "
                f"Multi-frequency stacking requires a shared angle grid."
            )
        if len(m) != n_angles or len(p) != n_angles:
            raise ConfigurationError(
                f"OASR: payload length mismatch at frequency index {i}: "
                f"angles={len(a)}, magnitude={len(m)}, phase={len(p)}"
            )
    if n_frequencies == 1:
        R = np.asarray(R_lists[0], dtype=float)
        phi_deg = np.asarray(phi_lists[0], dtype=float)
    else:
        R = np.column_stack([np.asarray(m, dtype=float) for m in R_lists])
        phi_deg = np.column_stack([np.asarray(p, dtype=float) for p in phi_lists])
    # OASR writes the phase as a principal value (`oases/src/oasjun21.f:102`
    # takes `atan2z`), but every consumer interpolates it linearly between
    # bracketing angles, which `misc/RefCoef.f90:119` states requires an
    # unwrapped phase: "Assumes phi has been unwrapped so that it varies
    # smoothly." Interpolating across a +-360 deg step sweeps the phase the long
    # way round and returns a reflection coefficient of the wrong sign. Unwrap
    # along the angle axis, which is a no-op for an already-smooth table.
    phi = np.unwrap(np.deg2rad(phi_deg), axis=0)
    return theta, R, phi, freqs


def _oasn_freq_axis(data: dict) -> np.ndarray:
    """Reconstruct the frequency vector from an OASN reader payload.

    The OASN ``.xsm`` / ``.rpo`` headers carry ``freq_min``, ``freq_max``,
    ``n_frequencies`` (and a ``freq_delta`` cross-check). We rebuild the
    axis with ``np.linspace`` since the spacing OASN advertises is
    equispaced between FREQ1 and FREQ2.
    """
    n = int(data.get('n_frequencies', 1))
    f1 = float(data.get('freq_min', 0.0))
    f2 = float(data.get('freq_max', f1))
    if n <= 1:
        return np.array([f1], dtype=float)
    return np.linspace(f1, f2, n, dtype=float)


def _warn_offset_ignored_under_auto_sampling(model_name, integration_offset,
                                             nw_samples, options=None,
                                             offset_letters='J',
                                             auto_sampling_zeroes_offset=True):
    """Warn that a caller's contour offset never reaches the integration.

    Two independent ways the binary discards it, either of which is enough:

    * **automatic wavenumber sampling** — ``unoast31.f:429`` sets
      ``OFFDB=0E0`` inside the ``NWVNOin < 0`` branch (``unoasp22.f:323`` for
      OASP), so ``:499-507`` then applies the kernel's own default and prints
      ``THE DEFAULT CONTOUR OFFSET IS APPLIED``. The offset still reaches the
      deck; the binary throws it away. The manual (``oast.tex:547-550``)
      documents only that ``IC1``/``IC2`` have no effect here, so following it
      is not enough to know this.
    * **an option line that never reads the token** — the frequency line is
      read with the offset field only under the letters in ``offset_letters``
      (``unoast31.f:126-133``, ``unoasp22.f:126-133``,
      ``unoassp30.f:135-142``). Without one of them the binary zeroes the
      offset *before* the read and consumes a shorter record, so the value
      uacpy wrote is not even parsed. 'J' also gates the application itself
      (``unoast31.f:499``, ``unoasp22.f:354-366``).

    OASN is unaffected by the first — ``unoasn22.f:283`` tests ``OFFDBIN``,
    which its automatic branch never touches — and passes
    ``auto_sampling_zeroes_offset=False`` to say so. It is *not* exempt from
    the second: ``:141-145`` reads the ``OFFDBIN`` token only under
    ``ICNTIN > 0``, which only ``'J'``/``'j'`` sets (``:673-675``).
    """
    if not integration_offset:
        return
    reasons = []
    if auto_sampling_zeroes_offset and nw_samples <= 0:
        reasons.append(
            f"automatic wavenumber sampling (nw_samples={nw_samples}) zeroes "
            f"the contour offset and applies the kernel's own default — pin "
            f"nw_samples >= 1 to use the value")
    if options is not None and not (set(str(options)) & set(offset_letters)):
        reasons.append(
            f"the option line for this run ({str(options)!r}) carries none of "
            f"{sorted(offset_letters)}, so the binary zeroes the offset and "
            f"reads a frequency line without it — add 'J' (complex "
            f"integration contour) to use the value")
    if not reasons:
        return
    warnings.warn(
        f"{model_name}(integration_offset={integration_offset:g}) has no "
        f"effect: " + "; ".join(reasons) + ". Drop integration_offset to "
        f"silence this.",
        UserWarning, skip_file_prefixes=USER_FRAME_SKIP)


def _reject_unintegrable_spectral_exponent(model_name, spectral_exponent):
    """Refuse a roughness exponent whose power spectrum does not converge.

    ``M <= 1.5`` leaves the spectrum unintegrable (``oassp.tex:356-362``; the
    exponent reaches ``amod(m) = fac(3+…)`` at ``oaseun31.f:99``). The deck
    writer refuses it too, but by then the scattering models have already
    spent the mean-field binary — and this depends on a constructor argument
    alone, so it belongs here.
    """
    if float(spectral_exponent) > 1.5:
        return
    raise ConfigurationError(
        f"{model_name}(spectral_exponent={float(spectral_exponent)}) must "
        f"exceed 1.5, or the roughness power spectrum is not integrable "
        f"(oassp.tex:356-362; the exponent reaches amod(m)=fac(3+…) at "
        f"oaseun31.f:99).",
        remediation="Pass spectral_exponent > 1.5 (2.0 is the default).",
    )


def _oases_find_executable(model: PropagationModel, name: str) -> Path:
    """Locate an OASES binary, preferring the ``<name>_bash`` wrapper.

    Searches ``uacpy/bin/oases``, ``uacpy/bin/oalib``, and
    ``uacpy/third_party/oases/bin``.
    """
    return model._find_executable_in_paths(
        [f'{name}_bash', name],
        bin_subdirs=['oases', 'oalib'],
        dev_subdir='oases',
    )


#: OASES' wavenumber-array bound, ``NPEXP = 16, NP = 2**NPEXP``
#: (third_party/oases/src/compar.f:37-38). OAST stops above it
#: (unoast31.f:459 ``IF (NWVNO.GT.NP) STOP '>>> TOO MANY WAVENUMBERS <<<'``),
#: which is what catches its automatic branch — a pinned NW never reaches
#: that test, being clamped by ``NWVNO=MIN0(NWVNO,NP)`` at :435 first.
#: OASP has no such test at all — AUTSAM (unoasp22.f:1130
#: ``NW=(WNMAX-WNMIN)/DK+1``) applies no clamp and :174's MIN0 runs before
#: it — so it runs past the arrays and writes a transfer function whose
#: values are meaningless.
_OASES_NP = 65536

#: OASP echoes the count it settled on before the frequency loop
#: (unoasp22.f:340, :349).
_OASES_NWVNO_ECHO = re.compile(r'NO\. OF WAVENUMBERS:\s*(\d+)')


def _oases_subprocess_env(base_name: str, **extras: str) -> dict:
    """Build the FORnnn env-var dict the OASES csh wrappers set.

    OPFILW/OPFILR/OPFILB resolve every Fortran unit through
    ``GETENV('FORnnn')`` (oases/src/oashun21.f:555-556, :590, :629), falling
    back to ``<deck stem>.<nnn>`` when the variable is unset (:560, :633) —
    so a unit with no entry here still lands on the base_name (e.g. the
    ``.021`` OASN and OASR leave behind), and only a WRITE that bypasses
    OPFIL entirely produces a compiler-named ``fort.NN``. The binary is
    therefore run directly and this dict stands in for
    ``third_party/oases/bin/{oast,oasn,oasp,oasr}``. The keys every
    sub-model needs are FOR001 (the ``.dat`` deck), FOR019/FOR020 (the
    ``.plp`` header and ``.plt`` data OASES' plotters write), FOR028/FOR029
    (contour output, ``.cdr``/``.bdr`` in the wrappers, which uacpy never
    reads) and FOR045 (scattering right-hand sides, ``.rhs``). FOR045's
    ``.045`` suffix is load-bearing: it is the file the whole OASS/OASSP
    chain consumes — their ``_run_mean_field`` requires ``<stem>.045`` from
    the producer run and ``_execute`` re-points the consumer's FOR045 at
    that same name. ``extras``
    supplies the per-binary units the wrappers add (e.g. FOR002='src' for
    OAST, FOR016='xsm' for OASN) — pass the suffix without the
    ``base_name + '.'`` prefix. ``base_name`` is the stem of the input file
    (no extension).
    """
    import os
    # ``base_name`` is interpolated straight into the FORnnn filenames the
    # OASES csh wrappers open; a value with a path separator or ``..`` would
    # become a traversal. It is a hard-coded literal in every caller today —
    # keep it that way by rejecting anything that isn't a plain stem.
    if not re.fullmatch(r'[A-Za-z0-9_]+', base_name):
        raise ConfigurationError(
            f"OASES base_name must match [A-Za-z0-9_]+; got {base_name!r}"
        )
    env = os.environ.copy()
    env['FOR001'] = f'{base_name}.dat'
    env['FOR019'] = f'{base_name}.plp'
    env['FOR020'] = f'{base_name}.plt'
    env['FOR028'] = f'{base_name}.028'
    env['FOR029'] = f'{base_name}.029'
    env['FOR045'] = f'{base_name}.045'
    for key, suffix in extras.items():
        env[key] = f'{base_name}.{suffix}'
    return env


#: Roughness power spectra OASS and OASSP implement. GETOPT's only spectrum
#: letter is 'g'/'G' (goff, unoass21.f:674, unoassp30.f:1034); its absence
#: means Gaussian.
_OASS_SPECTRA = ('gaussian', 'goff-jordan')

#: RMS roughness (m) below which SCTRHS writes no boundary operators at all:
#: it skips any interface with ROUGH2 < 1e-10 (oaseun31.f:2310) and
#: ROUGH2 = ROUGH**2 (:379).
_OASS_MIN_MEAN_FIELD_ROUGHNESS = 1e-5

#: Deck layer 2 is the first water record, and ``ROUGH(M)`` is the roughness of
#: the interface at the TOP of layer M (``oaseun31.f:381-383``), so layer 2's
#: RG is the sea surface. ``SCTRHS`` stamps its records ``IN1 = 2``
#: (``oaseun31.f:2308-2309``), which is exactly what ``INTFC = 2`` selects.
_OASS_SURFACE_INTERFACE = 2

#: OASS echoes the wavenumber count it settled on after re-deriving it from
#: the .rhs (unoass21.f:231, FORMAT 550 at :50).
_OASS_NW_ECHO = re.compile(r'NW\s+IC1\s+IC2\s*\n\s*(\d+)')
#: FORMAT 550 (unoass21.f:50) prints NWVNO under 1X,I5: any count >= 100000
#: renders as asterisks, which is itself proof of a count far past NP.
_OASS_NW_I5_OVERFLOW = re.compile(r'NW\s+IC1\s+IC2\s*\n\s*\*+')


def _oass_wavenumber_echo(process) -> Optional[int]:
    """``NWVNO`` OASS ran with, from its own stdout.

    Under the reverberation options the deck's Block VII is discarded and the
    count comes from the mean field's wavenumber step (``unoass21.f:209-215``),
    so the binary's echo is the only place the value it used appears.
    """
    stdout = getattr(process, 'stdout', '') or ''
    match = _OASS_NW_ECHO.search(stdout)
    if match:
        return int(match.group(1))
    if _OASS_NW_I5_OVERFLOW.search(stdout):
        # The I5 field overflowed: the count is >= 100000, far past
        # NP = 65536 — return a value the overrun check trips on rather
        # than degrading to silence exactly when the overrun is worst.
        return 100000
    return None


#: The bare ``fort.46`` an option-``'s'`` run leaves in the work dir.
#: ``oaseun31.f:1900`` and ``:2007`` WRITE unit 46 with no ``OPFILW`` (the
#: call is commented out at ``:2012``), so the write lands on the default
#: name instead of a ``base_name`` suffix. Declared by the classes whose
#: binary can reach those writes, so a pinned ``work_dir`` starts each run
#: clean rather than carrying a previous run's file forward.
#:
#: Not hoisted onto :class:`OASES`, because the list is a census of what each
#: binary writes and two of the six do not write this one. ``OASS`` cannot
#: reach the writes at all: its consumer deck carries no ``'s'``, so
#: ``SCTOUT`` stays at the ``.FALSE.`` ``unoass21.f:721-722`` initialises it
#: to, and the ``fort.46`` seen after an OASS run is the OAST producer's,
#: cleared under that class. ``OASSP`` stages units 45/46 on the mean-field
#: stem (``<mean_stem>.045`` / ``.046``), which are different files from the
#: bare name, so it neither writes nor reads this one. A base-class entry
#: would be inert for both — it would just stop saying which binary writes
#: what.
_SCTOUT_BARE_FORT46 = ('fort.46',)


class OASES(PropagationModel):
    """Public base + factory for the OASES sub-models
    (OAST/OASN/OASR/OASP/OASS/OASSP).

    Each sub-model wraps one OASES binary and differs only in its per-binary
    ``FORnnn`` unit-number map (``_FOR_FILES``) and its run modes; all share the
    one ``_execute`` below. They subclass ``OASES``, so ``isinstance(model,
    OASES)`` is True for any of them. ``OASES`` itself is abstract — instantiate
    a sub-model directly, or use :meth:`OASES.for_mode` to pick one by
    ``RunMode``."""

    _FOR_FILES: dict = {}

    # Files this sub-model's binary writes: suffixes appended to
    # ``base_name``, plus the bare ``fort.NN`` names some OASES builds emit
    # instead. Both are cleared before launch so a pinned ``work_dir``
    # cannot return an earlier run's output as this run's answer.
    _OUTPUT_SUFFIXES: tuple = ()
    _OUTPUT_FORT_FILES: tuple = ()

    def __new__(cls, *args, **kwargs):
        if cls is OASES:
            raise ConfigurationError(
                "OASES is the abstract base for "
                "OAST/OASN/OASR/OASP/OASS/OASSP. "
                "Instantiate a sub-model directly, or use "
                "OASES.for_mode(run_mode=...) to pick one by RunMode."
            )
        return super().__new__(cls)

    @classmethod
    def for_mode(cls, run_mode: RunMode = RunMode.COHERENT_TL, *,
                 broadband: bool = False, reverberation: bool = False,
                 **kwargs) -> PropagationModel:
        """Instantiate the OASES sub-model that handles ``run_mode``.

        ``COHERENT_TL`` → ``OAST`` (``OASP`` when ``broadband=True``);
        ``COVARIANCE`` → ``OASN`` (``OASS`` when ``reverberation=True``);
        ``REPLICA`` → ``OASN``; ``REFLECTION`` → ``OASR``;
        ``BROADBAND``/``TIME_SERIES`` → ``OASP`` (``OASSP`` when
        ``reverberation=True``, which returns the scattered field rather than
        the coherent one); ``REVERBERATION`` → ``OASS``, which needs no flag
        because nothing else emits it. ``**kwargs`` forward to the chosen
        sub-model (a kwarg it doesn't accept raises ``TypeError``).
        """
        dispatch = {
            RunMode.COHERENT_TL: OASP if broadband else OAST,
            RunMode.COVARIANCE: OASS if reverberation else OASN,
            RunMode.REPLICA: OASN,
            RunMode.REFLECTION: OASR,
            RunMode.BROADBAND: OASSP if reverberation else OASP,
            RunMode.TIME_SERIES: OASSP if reverberation else OASP,
            RunMode.REVERBERATION: OASS,
        }
        target = dispatch.get(run_mode)
        if target is None:
            raise UnsupportedFeatureError(
                'OASES', str(run_mode),
                alternatives=[str(m) for m in dispatch],
                alternatives_label='run modes',
            )
        return target(**kwargs)

    def _max_receiver_depth(self, env: Environment) -> float:
        """Deepest depth OASES resolves the field at: the full media stack.

        OASES meshes the sediment layers as media in their own right, so the
        field is defined through them and a source or receiver inside the
        seabed is a supported geometry — buried and sub-bottom sources are a
        large part of what the seismo-acoustic family is for.
        """
        return self._total_media_depth(env)

    def _wavenumber_echo_scale(self) -> int:
        """Factor between the echoed wavenumber count and the count actually
        integrated. 1 unless the sub-model rewrites ``NWVNO`` after printing
        it (see :meth:`OASSP._wavenumber_echo_scale`)."""
        return 1

    def _reject_wavenumber_overrun(self, process) -> None:
        """Raise when the run integrated on more wavenumbers than NP holds.

        Under automatic sampling AUTSAM sizes the wavenumber axis from the
        frequency-range product with no bound (unoasp22.f:1130 and its OASSP
        copy unoassp30.f:1125, ``NW=(WNMAX-WNMIN)/DK+1``), and — unlike OAST,
        which stops at unoast31.f:459 — neither OASP nor OASSP has an
        ``NWVNO.GT.NP`` test (their ``MIN0(NWVNO,NP)`` runs *before* AUTSAM,
        so it cannot bound the automatic branch). Past NP the run completes,
        exits 0 and writes a full-size ``.trf`` holding numbers that are not
        the field (measured 3.2e27 against 1.3e-4 for the same geometry with
        NW pinned under NP). The count the binary settled on is in its own
        stdout, so read it from there rather than re-deriving AUTSAM.
        """
        scale = self._wavenumber_echo_scale()
        counts = [scale * int(m) for m in
                  _OASES_NWVNO_ECHO.findall(getattr(process, 'stdout', '') or '')]
        if not counts or max(counts) <= _OASES_NP:
            return
        raise ConfigurationError(
            f"{self.model_name} sampled {max(counts)} wavenumbers, past "
            f"OASES' array bound NP = {_OASES_NP} (oases/src/compar.f:37-38); "
            f"this binary has no bound check on automatic sampling, so the "
            f"transfer function it wrote is not the acoustic field. The "
            f"count grows with the highest frequency times the largest "
            f"receiver range.",
            remediation=(
                "Shorten receiver.ranges, lower freq_max / the source "
                "frequency, or pin nw_samples <= "
                f"{_OASES_NP} to integrate on a bounded grid."),
        )

    def _execute(self, base_name: str, work_dir: Path, *, extra_env=None):
        """Run the OASES binary and return its ``CompletedProcess``.

        FOR005 stays as stdin per OASES docs. The completed process is
        returned rather than dropped: OASES writes no print file, so its
        stdout/stderr are the only record of what it did.

        ``extra_env`` adds FORnnn unit assignments that cannot live in
        ``_FOR_FILES``: ``_oases_subprocess_env`` derives every suffix from
        this run's ``base_name``, but the OASS/OASSP scattering
        post-processors read files named after the *mean-field* run's stem
        (an output for the producer, an input here). ``stale_outputs``
        clears this run's stem only, so the producer's files survive the
        pre-launch wipe — the two stems are distinct by construction.
        """
        env = _oases_subprocess_env(base_name, **self._FOR_FILES)
        if extra_env:
            env.update(extra_env)
        for name in self._OUTPUT_FORT_FILES:
            Path(work_dir).joinpath(name).unlink(missing_ok=True)
        result = self._run_and_attach_prt(
            [str(self._exe)], work_dir, base_name, env=env,
            stale_outputs=self._OUTPUT_SUFFIXES,
        )
        self._warn_on_stdout_warnings(result)
        return result

    def _warn_on_stdout_warnings(self, result) -> None:
        """Surface the binary's own non-fatal warnings from stdout.

        The base class's pass reads ``<base_name>.prt``, which the Acoustics
        Toolbox writes and OASES does not — so without this, six model classes
        drop every diagnostic their binary emits. The one this was measured on
        is the only check of its kind anywhere in the pipeline: an elastic
        half-space with ``(cs/cp)**2 > 0.75`` makes ``oaseun31.f:204-207``
        print ``>>>>>WARNING: UNPHYSICAL SPEED RATIO``, and nothing on the
        Python side tests that ratio.

        Matched narrowly: OASES prefixes ordinary progress traces with the
        same ``>>>`` (``>>> Entering SOLDIS <<<``, ``>>> Done, CPU=``), so
        only a ``>``-prefixed line that also says *warning* is forwarded.
        These are the solver's words, passed through verbatim rather than
        reinterpreted — the same contract as the ``.prt`` pass. Fatals are
        already raised by ``_raise_on_fortran_fatal`` and quoted by
        ``_require_output``.
        """
        text = getattr(result, 'stdout', None)
        if not text:
            return
        seen, lines = set(), []
        for raw in text.splitlines():
            line = raw.strip()
            if (line.startswith('>') and 'warning' in line.lower()
                    and line not in seen):
                seen.add(line)
                lines.append(line)
        if lines:
            joined = "\n  ".join(lines)
            warnings.warn(
                f"{self.model_name} reported {len(lines)} non-fatal "
                f"warning(s) on stdout:\n  {joined}",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)


class OAST(OASES):
    """
    OAST - OASES Transmission Loss Model

    Computes transmission loss using wavenumber integration. Best for
    range-independent environments with depth-dependent sound speed profiles.

    Parameters
    ----------
    executable : Path, optional
        Path to OAST binary. Auto-detected if ``None``.
    compute_contour : bool, optional
        Add ``'C'`` option (range-depth contour plot). Effective default
        ``False``.
    compute_depth_average : bool, optional
        Add ``'A'`` option (depth-averaged TL). Effective default ``False``.
    complex_contour : bool, optional
        ``'J'`` option (complex integration contour). Effective default
        ``True``.
    options : str, optional
        Raw OASES options string (e.g. ``'N J T C'``), written verbatim;
        ``None`` derives it from ``compute_contour`` /
        ``compute_depth_average`` / ``complex_contour``. Combining a raw
        string with any of those three flags raises
        ``ConfigurationError`` — the string replaces the whole option
        line, so a flag passed alongside it would be discarded.
    integration_offset : float
        Wavenumber-integration contour offset (dB/wavelength). Default 0.
    nw_samples : int
        Number of wavenumber samples; ``-1`` lets OASES choose.
    c_low, c_high : float, optional
        ``CMIN``/``CMAX`` (m/s), Block VII's phase-speed window
        (``unoast31.f:213``). ``None`` derives the pair from the water
        column alone (:func:`oases_wavenumber_bounds`), which is what an
        elastic seabed needs widening: its shear and interface branches have
        phase speeds *below* the slowest water speed, so they fall outside
        the derived ``k_max`` and never enter the integral. ``c_low``
        admits them; ``c_high`` caps the evanescent tail.
    plot_rmin, plot_rmax : float, optional
        TL plot range axis bounds (m); ``None`` → 0 /
        ``receiver.range_max``.
    vrec : float
        Receiver velocity (m/s) for OAST's source/receiver **dynamics**
        option, the fifth token of the frequency line. The binary reads it
        only when the option string carries lowercase ``'d'``
        (``unoast31.f:125-127``), which the derived option line never
        emits — so a non-zero ``vrec`` without a raw ``options`` string
        containing ``'d'`` raises ``ConfigurationError`` (the same
        contract as ``dip_angle``, whose value is read only under
        ``'4'``). Default 0.

        Despite the binary's own "Doppler compensation" wording
        (``unoast31.f:1094``), ``oast.tex:215-222`` is explicit that source
        and receiver move *at the same speed and direction*, so **there is
        no Doppler shift** — only a Green's function different from the
        static one. OASP's ``'d'`` is the real radial-Doppler option and
        carries three tokens (``IT VS VR``, ``oasp.tex:227-234``).
    dip_angle : float, optional
        Fault dip angle (degrees) for the dip-slip moment source that
        option ``'4'`` selects (``unoast31.f:1117-1122``). INSRC reads it
        from the source record — as the second token without ``'L'``
        (``oaseun31.f:1114``), as the seventh with it
        (``oaseun31.f:1089``). ``None`` writes 0 when ``'4'`` is present
        and raises when it is not.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Range-independent wavenumber integration; consumes layered seabed
    columns natively. **Collapse defaults (overrides of
    :data:`DEFAULT_COLLAPSE`).** Per-model: ``'ssp': 'mean'``,
    ``'bottom_range': 'median'`` (the layer stack is kept).

    ``COHERENT_TL`` here returns a **real dB Field** — ``kind='pressure'``,
    ``unit='dB'``, real dtype (transmission loss is the ``unit`` axis of a
    pressure field, not a registered ``kind``) — where :class:`OASP` returns
    complex Pa pressure for the same run mode: OAST's ``.plt`` carries only
    real TL, so there is no complex pressure to hand back and no phase
    reference to tag. Use :class:`OASP` when the consumer needs complex
    pressure — coherent array processing, phase, or off-grid receiver
    ranges through sharp interference nulls.

    Examples
    --------
    >>> from uacpy.models import OAST
    >>> oast = OAST()
    >>> result = oast.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OAST:
    # range-independent wavenumber integration; multi-layer fluid + elastic
    # bottom honoured. Single spectral solve → mean SSP / median bottom column.
    spec = ModelSpec(
        modes=(RunMode.COHERENT_TL,),
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
    )
    source = 'oases'

    def __init__(
        self,
        executable: Optional[Path] = None,
        compute_contour: Optional[bool] = None,
        compute_depth_average: Optional[bool] = None,
        complex_contour: Optional[bool] = None,
        options: Optional[str] = None,
        integration_offset: float = 0.0,
        nw_samples: int = -1,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        plot_rmin: Optional[float] = None,
        plot_rmax: Optional[float] = None,
        vrec: float = 0.0,
        dip_angle: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        # Kept as passed so ``copy()`` round-trips and so an explicit flag
        # alongside a raw ``options`` string is distinguishable from the
        # default. Resolved by ``_resolve_options``.
        self.compute_contour = compute_contour
        self.compute_depth_average = compute_depth_average
        self.complex_contour = complex_contour
        # Raw OASES option string (e.g. ``'N J T C'``). ``None`` lets the
        # wrapper derive it from compute_contour / compute_depth_average /
        # complex_contour; a raw string replaces the deck's option line
        # outright, so the two ways of specifying it are exclusive.
        self.options = options
        if options is not None:
            pinned = [n for n in ('compute_contour', 'compute_depth_average',
                                  'complex_contour')
                      if getattr(self, n) is not None]
            if pinned:
                raise ConfigurationError(
                    f"OAST: options={options!r} replaces the whole option "
                    f"line, so {', '.join(pinned)} would be discarded "
                    f"silently. Pass either the raw string or the typed "
                    f"flags, not both — note the derived string includes "
                    f"'J' (complex integration contour) by default, which a "
                    f"raw string must repeat to keep."
                )
        self.integration_offset = float(integration_offset)
        # ``-1`` lets the OASES kernel pick its own wavenumber sample count.
        self.nw_samples = int(nw_samples)
        # Block VII's CMIN/CMAX. ``None`` leaves the writer's water-column
        # derivation in place; a value replaces that side of the window.
        self.c_low = float(c_low) if c_low is not None else None
        self.c_high = float(c_high) if c_high is not None else None
        # OAST reads the offset token under 'J', 'O' (complex frequency) or
        # 'd' (dynamics) — unoast31.f:126-133 — and applies it only under 'J'
        # (:499). The derived line carries 'J' unless complex_contour=False.
        _warn_offset_ignored_under_auto_sampling(
            'OAST', self.integration_offset, self.nw_samples,
            options=self._resolve_options(), offset_letters='JOd')
        # Plot-axis bounds in metres. ``None`` → 0 / receiver.range_max.
        self.plot_rmin = float(plot_rmin) if plot_rmin is not None else None
        self.plot_rmax = float(plot_rmax) if plot_rmax is not None else None
        # VREC, the fifth frequency-line token OAST reads only under the
        # lowercase 'd' (dynamics) option. The derived option line never
        # carries 'd', so a value given without a raw options string that
        # does would be written into a 4-token line the binary stops
        # reading after COFF (unoast31.f:125-127) — rejected here the way
        # dip_angle is rejected without '4'.
        self.vrec = float(vrec)
        if self.vrec and 'd' not in set(self._resolve_options()) - set(' \t\n'):
            raise ConfigurationError(
                f"OAST(vrec={self.vrec:g}) cannot reach the binary: INFREQ "
                f"reads the fifth frequency-line token only under the "
                f"lowercase 'd' option (unoast31.f:125-127), and the option "
                f"line for this run ({self._resolve_options()!r}) does not "
                f"carry it.",
                remediation=("Pass a raw options string with 'd' (e.g. "
                             "options='N J T d') alongside vrec, or drop "
                             "vrec."),
            )
        # DA, the dip angle INSRC reads only under the '4' (dip-slip) option.
        self.dip_angle = float(dip_angle) if dip_angle is not None else None

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable, lambda: _oases_find_executable(self, 'oast'),
        )

    def _resolve_options(self) -> str:
        """The OASES option line for this run.

        A raw ``options`` string is written verbatim; otherwise the letters
        are derived from the typed flags on top of ``'N T'`` (normal stress
        + TL table). The constructor rejects the two being combined.
        """
        if self.options is not None:
            return self.options
        opt = ['N', 'T']
        if self.complex_contour is None or self.complex_contour:
            opt.append('J')
        if self.compute_contour:
            opt.append('C')
        if self.compute_depth_average:
            opt.append('A')
        return ' '.join(opt)

    def _reject_multi_frequency_source(self, source: Source) -> None:
        """Refuse a frequency sweep, naming the model that does compute one.

        The deck and the reader both handle ``NFREQ > 1``: OAST writes one TL
        curve per plotted receiver per frequency and ``read_oast_tl`` returns
        an ``(n_freq, n_depths, n_ranges)`` stack. What cannot follow is the
        :class:`~uacpy.core.results.Field`. OAST rebuilds its range grid
        *inside* the frequency loop, so ``DX`` scales as ``1/f`` and the
        sweep has one range axis per frequency — the reader returns
        ``ranges`` as ``(n_freq, n_ranges)`` for exactly that reason — while
        a Field carries a single ``'range'`` coordinate.

        Putting them on a common axis would mean interpolating dB, which is
        all the ``.plt`` carries: no complex pressure exists on disk to
        interpolate instead. That is the smearing this model already warns
        about and tells callers to avoid, and here it would be unavoidable
        rather than a choice, since at most one frequency's native grid can
        be the shared one.

        :class:`OASP` is the multi-frequency wavenumber-integration product:
        it writes complex pressure to a ``.trf`` on one caller-anchored
        range grid and returns a ``(depth, range, frequency)`` Field. So the
        sweep is refused here rather than approximated.

        ``validate_inputs`` would also refuse this (``COHERENT_TL`` is in
        ``_SINGLE_FREQUENCY_MODES``), but its shared remedy is
        ``RunMode.BROADBAND``, which OAST does not implement — following it
        lands on ``UnsupportedFeatureError`` naming no alternative at all.
        """
        freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
        if freqs.size <= 1:
            return
        raise UnsupportedFeatureError(
            'OAST',
            f"a {freqs.size}-frequency source "
            f"({freqs.min():g}-{freqs.max():g} Hz) — OAST rebuilds its range "
            f"grid inside the frequency loop (DX proportional to 1/f), so a "
            f"sweep has one range axis per frequency and no single Field can "
            f"carry it; and the .plt holds only real TL, so there is no "
            f"complex pressure to resample onto a common axis without "
            f"smearing interference nulls",
            alternatives=['OASP (complex pressure on one range grid, '
                          'returns a (depth, range, frequency) Field)'],
        )

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """
        Run OAST transmission loss computation

        Parameters
        ----------
        env : Environment
            Ocean environment (must be range-independent)
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array

        Returns
        -------
        result : Result
            Transmission loss field
        """
        self._require_run_triple(env, source, receiver)
        self._reject_unsupported_run_kwargs(
            frequencies=frequencies, source_waveform=source_waveform,
            sample_rate=sample_rate, output_duration=output_duration)
        run_mode = self._resolve_run_mode(run_mode)
        self._reject_multi_frequency_source(source)

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        fm = self._setup_file_manager()

        try:
            base_name = 'oast_run'
            input_file = fm.get_path(f'{base_name}.dat')

            user_options = self._resolve_options()

            self._log(f"Writing OAST input file: {input_file} (options={user_options})")
            writer_kwargs: dict = {
                'integration_offset': self.integration_offset,
                'nw_samples': self.nw_samples,
                'vrec': self.vrec,
            }
            if self.c_low is not None:
                writer_kwargs['c_low'] = self.c_low
            if self.c_high is not None:
                writer_kwargs['c_high'] = self.c_high
            if self.plot_rmin is not None:
                writer_kwargs['plot_rmin'] = self.plot_rmin
            if self.plot_rmax is not None:
                writer_kwargs['plot_rmax'] = self.plot_rmax
            if self.dip_angle is not None:
                writer_kwargs['dip_angle'] = self.dip_angle
            write_oast_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                options=user_options,
                **writer_kwargs,
            )

            proc = self._execute(base_name, fm.work_dir)

            # Read output - OAST creates .plt (data) and .plp (metadata) files
            # Per OASES documentation: FOR019 -> .plp, FOR020 -> .plt
            output_file = self._require_output(
                [fm.get_path(f'{base_name}.plt')],
                what='a TL table (FOR020 .plt)', process=proc,
                hint=('Compare against the reference decks in '
                      'third_party/oases/tloss/.'),
            )

            self._log(f"Reading OAST output: {output_file}")
            oast_out = read_oast_tl(
                filepath=output_file,
                receiver_depths=receiver.depths,
            )
            tl_data = oast_out['tl']
            native_depths = oast_out['depths']
            native_ranges = oast_out['ranges']
            metadata = oast_out['metadata']

            kw = self._result_kwargs(
                source,
                backend='oast',
                frequencies=float(np.atleast_1d(source.frequencies)[0]),
            )
            kw['metadata'].update(metadata)
            native = Field(
                data=tl_data,
                coords={'depth': native_depths, 'range': native_ranges},
                **kw,
            )
            receiver_ranges = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
            receiver_depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
            ranges_match = (
                len(native_ranges) == len(receiver_ranges)
                and np.allclose(native_ranges, receiver_ranges)
            )
            if ranges_match:
                result = native
            else:
                # OAST writes only real TL (dB) to .plt — there is no complex
                # pressure on disk — so off-grid receiver ranges are obtained
                # by linear interpolation IN dB. That over-fills sharp
                # interference minima (a true −80 dB null between −40 dB
                # neighbours reads back ≈ −50 dB). Align receiver.ranges to the
                # OAST native FFT grid (ranges_match) to avoid the smoothing,
                # or use OASP (.trf complex pressure) when null depth matters.
                warnings.warn(
                    "OAST: receiver.ranges do not match the OAST native FFT "
                    "range grid, so TL is linearly interpolated IN dB. This "
                    "smears sharp interference minima (a true −80 dB null "
                    "between −40 dB neighbours reads back ≈ −50 dB). Align "
                    "receiver.ranges to the native grid, or use OASP (.trf "
                    "complex pressure) when null depth matters.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
                result = native.resample_to(
                    ranges=receiver_ranges, depths=receiver_depths)
                result.metadata['oast_native_ranges'] = native_ranges
                result.metadata['interpolated'] = True
                # Interpolation cannot extrapolate: any requested range off
                # the native grid's hull comes back NaN, and the hull does
                # not start at r = 0 — the FFT grid's first sample sits one
                # range step out (measured 1.875 m on a 100-m Pekeris case).
                native_min = float(np.min(native_ranges))
                native_max = float(np.max(native_ranges))
                outside = ((receiver_ranges < native_min)
                           | (receiver_ranges > native_max))
                if outside.any():
                    offending = np.array2string(
                        receiver_ranges[outside], precision=4,
                        max_line_width=200)
                    warnings.warn(
                        f"OAST: {int(outside.sum())} receiver range(s) "
                        f"{offending} m lie outside the native FFT range "
                        f"grid ({native_min:g}-{native_max:g} m); dB "
                        f"interpolation cannot extrapolate, so those columns "
                        f"are NaN (no data). Move the receivers inside the "
                        f"native span — its first sample is at "
                        f"{native_min:g} m, not r = 0 — or use OASP, whose "
                        f".trf grid starts at receiver.ranges.min().",
                        UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                    )

            self._attach_output_paths(
                result, fm.work_dir, base_name,
                primary_files=(('plt_file', '.plt'),),
            )

            self._log("OAST simulation complete")
            return result

        finally:
            fm.finish()

    # The wrapper script third_party/oases/bin/oast also names FOR002 (the
    # external source array option 'l' reads) and FOR023 (the tabulated
    # reflection coefficients options 't'/'b' read, oaseun31.f:3727, :3757),
    # but uacpy rejects all three letters (_UNWRITTEN_OPTION_BLOCKS), so
    # those units are never opened and need no entries here.
    _FOR_FILES: dict = {}
    # '.045' is the option-'s' boundary-operator dump (SCTOUT, unoast31.f:1073;
    # FOR045 is set for every run by _oases_subprocess_env). It is listed here
    # because it is an OASS/OASSP *input*: a pinned work_dir that kept a
    # previous run's .rhs would let _run_mean_field's _require_output accept it
    # as this run's, which is the reuse this list exists to close.
    _OUTPUT_SUFFIXES = ('.plt', '.plp', '.045')
    _OUTPUT_FORT_FILES = _SCTOUT_BARE_FORT46


class OASN(OASES):
    """
    OASN — OASES Noise, Covariance Matrices and Signal Replicas.

    Per the OASES manual (``third_party/oases/doc/oasn.tex``) OASN produces
    two frequency-domain array products:

    - **Covariance matrices** (option ``N`` → ``.xsm``): hydrophone ×
      hydrophone correlation per frequency. Used for ambient-noise
      characterisation or as an MFP measurement covariance.
    - **Replica fields** (option ``R`` → ``.rpo``): array response per
      candidate source position per frequency. Used as MFP templates.

    For depth-eigenfunction normal modes use :class:`Kraken`.

    Parameters
    ----------
    executable : Path, optional
        Path to OASN binary. Auto-detected if ``None``.
    options : str, optional
        Custom OASES options string; ``None`` derives it from the run
        mode (``N`` for COVARIANCE, ``R`` for REPLICA).
    surface_noise_level : float
        Surface-generated noise spectral level (dB re 1 µPa²/Hz),
        Block VI. OASES disables the source when ``abs(level) < 0.01``
        (``oasnun22.f:183``); a level at or below ``-0.01`` is not "off" —
        it names the Fortran unit number of a source-spectrum file uacpy
        does not write, so it is rejected. Requires a vacuum or air
        ``env.surface`` (``oasnun22.f:187``).
    white_noise_level : float, optional
        Uncorrelated (white) noise spectral level per hydrophone
        (dB re 1 µPa²/Hz), added to every covariance-matrix diagonal
        element. OASES has no off switch for this field: it adds
        ``10**(dB/10)`` to the diagonal unconditionally
        (``oasnun22.f:228``, ``:1157``), so an explicit ``0.0`` is a
        literal 0 dB — unit linear power — per sensor. ``None`` (the
        default) writes -200 dB, whose 1e-20 linear power is
        numerically nil against any noise field.
    deep_noise_level : float
        Deep broad-area source spectral level (dB re 1 µPa²/Hz). OASES
        uses two thresholds that disagree at exactly 0.01: the source
        radiates only for ``level > 0.01`` (``oasnun22.f:233`` skips it on
        ``.LE.0.01``) while Block VIII is read for ``level >= 0.01``
        (``oasnun22.f:324``). Unlike the surface level, a negative value
        simply disables it — the deep source has no spectrum-file form.
    deep_source_depth : float, optional
        Depth (m) of the deep broad-area noise source sheet; ``None``
        → half the water depth. Only written when ``deep_noise_level``
        is non-zero.
    discrete_sources : list of dict, optional
        Point sources; each dict may carry ``'depth'`` (m), ``'x'``
        (m), ``'y'`` (m) and ``'level'`` (dB, non-negative) — the four
        fields OASES reads (``oasnun22.f:380``). Any other key raises
        ``ConfigurationError``; OASES has no per-source phase.
    xmin, xmax : float, optional
        Replica candidate-grid x bounds (m); ``None`` → OASES defaults
        (100 / 10000).
    nx : int
        Number of replica grid points in x. Default 50.
    ymin, ymax : float, optional
        Replica candidate-grid y bounds (m); ``None`` → 0 / 0.
    ny : int
        Number of replica grid points in y. Default 1.
    zmin, zmax : float, optional
        Replica candidate-grid depth bounds (m); ``None`` → 10 /
        ``env.depth - 10``.
    nz : int
        Number of replica grid points in depth. Default 20.
    c_low, c_high : float, optional
        Phase-speed bounds (m/s) for the wavenumber integrations,
        applied to both the noise and replica blocks; ``None`` →
        ``0.95 · min(c_water)`` and ``1e8``.
    integration_offset : float
        Wavenumber-integration contour offset (dB/wavelength). Default 0.
    nw_samples : int
        Number of wavenumber samples; ``-1`` lets OASES choose.
    plot_rmin, plot_rmax : float, optional
        TL plot range axis bounds (m).
    vrec : float
        Vertical receiver velocity (m/s) for Doppler. Default 0.
    offdb : float, optional
        Single-mode horizontal offset (dB).
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Supported run modes: ``RunMode.COVARIANCE`` (``.xsm`` →
    :class:`Covariance`) and ``RunMode.REPLICA`` (``.rpo`` →
    :class:`Replicas`). The convenience methods
    ``compute_covariance(...)`` / ``compute_replicas(...)`` route to
    these.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom_range': 'median'`` (the
    layer stack is kept).

    Examples
    --------
    >>> from uacpy.models import OASN
    >>> # A covariance needs a noise field: with no source configured the
    >>> # matrix is only the -200 dB white-noise floor (diagonal 1e-20).
    >>> oasn = OASN(surface_noise_level=40.0)
    >>> cov = oasn.compute_covariance(env, source, receiver)
    >>> # cov.covariance has shape (n_frequencies, n_rcv, n_rcv)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASN:
    # range-independent covariance / replica field; multi-layer bottom
    # honoured. Single spectral solve per frequency → mean SSP / median column.
    spec = ModelSpec(
        modes=(RunMode.COVARIANCE, RunMode.REPLICA),
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
    )
    source = 'oases'

    def __init__(
        self,
        executable: Optional[Path] = None,
        # Output / option control
        options: Optional[str] = None,
        # Noise field (Block VI): broad-area sources expressed as
        # spectral levels (dB re 1 µPa²/Hz) at three depths.
        surface_noise_level: float = 0.0,
        white_noise_level: Optional[float] = None,
        deep_noise_level: float = 0.0,
        deep_source_depth: Optional[float] = None,
        # Discrete (point) sources — list of dicts; each may carry
        # 'depth' (m), 'x' (m), 'y' (m), 'level' (dB).
        discrete_sources: Optional[list] = None,
        # Replica candidate-position grid (Block X). x/y/z in metres on
        # the public API, converted to km at write time. ``None`` lets
        # the writer apply OASES defaults (10 / depth-10 in z, 100 /
        # 10000 in x, 0 / 0 in y).
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        nx: int = 50,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
        ny: int = 1,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        nz: int = 20,
        # Phase-speed bounds for the wavenumber integrations (m/s).
        # Applied identically to OASN Block VIII (noise / discrete
        # sources) and Block X (replica generator). ``None`` → writer
        # derives c_water_min * 0.95 (c_low) and 1e8 (c_high).
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        # Wavenumber-axis sampling & TL plot axes.
        integration_offset: float = 0.0,
        nw_samples: int = -1,
        plot_rmin: Optional[float] = None,
        plot_rmax: Optional[float] = None,
        # Frequency-line extras: vrec (m/s, vertical receiver velocity
        # for Doppler), offdb (single-mode horizontal offset).
        vrec: float = 0.0,
        offdb: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        self.options = options
        self.surface_noise_level = float(surface_noise_level)
        # None means "no white noise": the writer maps it to the -200 dB
        # level whose linear power is numerically nil (OASES itself has no
        # off switch for the diagonal term).
        self.white_noise_level = (
            float(white_noise_level) if white_noise_level is not None else None
        )
        self.deep_noise_level = float(deep_noise_level)
        self.deep_source_depth = (
            float(deep_source_depth) if deep_source_depth is not None else None
        )
        self.discrete_sources = list(discrete_sources) if discrete_sources else None
        self.xmin = xmin
        self.xmax = xmax
        self.nx = int(nx)
        self.ymin = ymin
        self.ymax = ymax
        self.ny = int(ny)
        self.zmin = zmin
        self.zmax = zmax
        self.nz = int(nz)
        self.c_low = c_low
        self.c_high = c_high
        self.integration_offset = float(integration_offset)
        self.nw_samples = int(nw_samples)
        if self.nw_samples > 0:
            # On OASN a pinned NW is not just a density knob: it also widens
            # the INTEGRATION WINDOW. Automatic sampling calls AUTSMN
            # (oasnun22.f:432), which returns IC1/IC2 bracketing the
            # propagating band [2*pi*f/C2, 2*pi*f/C1] with 10% margins
            # (:1701-1712); the deck writer emits ``NW 1 NW`` for a pinned
            # count, and :438-440 takes ICUT1=1, ICUT2=NW verbatim. Since
            # :853-857 zeroes only the samples OUTSIDE [ICUT1, ICUT2], the
            # pinned deck integrates the entire window while the automatic
            # deck integrates the propagating band alone. Measured on a
            # 100 m Pekeris guide at 150 Hz, replica magnitudes then track NW
            # instead of converging: mean level -18.13, -14.67, -7.41, 0.00 dB
            # at NW = 2048, 4096, 8192, 16384 — no plateau. The default
            # (nw_samples=-1) is the cross-validated path: it matches an
            # independent Kraken modal bank in
            # test_matched_field_from_field.py.
            warnings.warn(
                f"OASN(nw_samples={self.nw_samples}): a pinned wavenumber "
                f"count also sets ICUT1=1, ICUT2=NW, which widens the "
                f"integration window from the propagating band that "
                f"automatic sampling selects to the full wavenumber axis. "
                f"Replica and covariance magnitudes then depend on NW rather "
                f"than converging with it. Pass nw_samples=-1 (the default) "
                f"unless you are deliberately integrating the whole axis.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        # OASN's deck has no range-axis or receiver-velocity field: it emits
        # covariance matrices and replicas, not TL-vs-range. ``unoasn22.f``
        # does recognise a 'D'/'d' option and carries a ``vrec`` variable
        # (:702-704, defaulted at :762), but nothing reads a value for it from
        # the input file — it stays at its 15.0 data-statement default, so the
        # argument cannot reach the binary whatever it is set to.
        # Accepting these silently would advertise knobs that cannot reach the
        # binary, so they are rejected at construction instead.
        for name, value in (('plot_rmin', plot_rmin), ('plot_rmax', plot_rmax)):
            if value is not None:
                raise ConfigurationError(
                    f"OASN({name}=...) has no effect: OASN writes covariance "
                    f"matrices and signal replicas, not a range-axis plot "
                    f"block.",
                    remediation=("Use OAST for transmission loss over a range "
                                 "axis, or drop the argument."),
                )
        if vrec:
            raise ConfigurationError(
                "OASN(vrec=...) has no effect: unoasn22.f:702-704 recognises "
                "the 'd' option and carries a vrec variable, but never reads a "
                "value for it from the deck — it keeps the 15.0 m/s default at "
                ":762, so the argument cannot reach the binary.",
                remediation="Use OAST with the 'd' option, or drop vrec.",
            )
        self.plot_rmin = None
        self.plot_rmax = None
        self.vrec = 0.0
        # Same OASES field as integration_offset (unoasn22.f:142 reads
        # FREQ1 FREQ2 NFREQ OFFDBIN); an explicit value wins in the writer.
        self.offdb = float(offdb) if offdb is not None else None
        # run() writes ``self.options or 'J'`` plus the run-mode letter, so
        # only a custom string can drop 'J' — and without it the binary
        # zeroes the offset and reads a three-token frequency line, the same
        # silent discard every sibling model warns about.
        _warn_offset_ignored_under_auto_sampling(
            'OASN', self.integration_offset, self.nw_samples,
            options=(self.options if self.options is not None else 'J'),
            auto_sampling_zeroes_offset=False)

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable, lambda: _oases_find_executable(self, 'oasn2_bin'),
        )

    def _has_noise_field(self) -> bool:
        """True when any Block VI source was configured on this instance.

        White noise counts whenever it was set at all: an explicit 0.0 is
        a literal 0 dB diagonal term, not "off", so the test is against
        ``None`` rather than truthiness.
        """
        return bool(self.surface_noise_level
                    or self.white_noise_level is not None
                    or self.deep_noise_level or self.discrete_sources)

    def _build_writer_kwargs(self) -> dict:
        """Materialise the writer's expected kwarg dict from ``self.*``.

        Translates uacpy's public API (singular names, metres) to the
        OASES file format the writer consumes (plural names for
        cmins/cmaxs, km for x/y axes). Only keys whose stored value is
        non-default are emitted, so the writer falls back to its own
        defaults for everything the user did not set.
        """
        kw: dict = {
            'integration_offset': self.integration_offset,
            'nw_samples': self.nw_samples,
            'surface_noise_level': self.surface_noise_level,
            'white_noise_level': self.white_noise_level,
            'deep_noise_level': self.deep_noise_level,
        }
        if self.deep_source_depth is not None:
            kw['deep_source_depth'] = self.deep_source_depth
        if self.offdb is not None:
            kw['offdb'] = self.offdb

        # Phase-speed bounds applied identically to OASN's noise and
        # replica integrations (singular on the public API, plural-with-
        # suffix in the writer per OASES's CMINS/CMAXS variable names).
        if self.c_low is not None:
            # Under its own name for the surface/deep *noise* blocks, and
            # under the writer's CMINS-style names for the discrete-source and
            # replica integrations — the docstring promises all four.
            kw['c_low'] = self.c_low
            kw['cmins_discrete'] = self.c_low
            kw['cmins_replica'] = self.c_low
        if self.c_high is not None:
            kw['c_high'] = self.c_high
            kw['cmaxs_discrete'] = self.c_high
            kw['cmaxs_replica'] = self.c_high

        # Replica grid x/y in km on disk; z in metres. nx/ny/nz always
        # forwarded so the writer reuses uacpy defaults rather than its
        # own OASES-doc ones when the user did not override.
        kw['replica_nx'] = self.nx
        kw['replica_ny'] = self.ny
        kw['replica_nz'] = self.nz
        if self.xmin is not None:
            kw['replica_xmin'] = float(m_to_km(self.xmin))
        if self.xmax is not None:
            kw['replica_xmax'] = float(m_to_km(self.xmax))
        if self.ymin is not None:
            kw['replica_ymin'] = float(m_to_km(self.ymin))
        if self.ymax is not None:
            kw['replica_ymax'] = float(m_to_km(self.ymax))
        if self.zmin is not None:
            kw['replica_zmin'] = self.zmin
        if self.zmax is not None:
            kw['replica_zmax'] = self.zmax

        # ``discrete_sources`` carry per-source x/y in metres (uacpy
        # API); OASES writes them in km (oasn.tex:343).
        if self.discrete_sources:
            converted = []
            for ds in self.discrete_sources:
                ds_km = dict(ds)
                if 'x' in ds_km:
                    ds_km['x'] = float(m_to_km(ds_km['x']))
                if 'y' in ds_km:
                    ds_km['y'] = float(m_to_km(ds_km['y']))
                converted.append(ds_km)
            kw['discrete_sources'] = converted

        return kw

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """
        Run OASN.

        Parameters
        ----------
        run_mode : RunMode, optional
            ``RunMode.COVARIANCE`` (default) → :class:`Covariance`.
            ``RunMode.REPLICA`` → :class:`Replicas` (build the OASN
            instance with ``xmin``/``xmax``/``zmin``/``zmax``/… already
            set on the constructor; sweeps go through ``model.copy(...)``).

        Returns
        -------
        Covariance or Replicas
        """
        self._require_run_triple(env, source, receiver)
        self._reject_unsupported_run_kwargs(
            frequencies=frequencies, source_waveform=source_waveform,
            sample_rate=sample_rate, output_duration=output_duration)
        run_mode = self._resolve_run_mode(run_mode)

        # OASN expresses the frequency axis as (fmin, fmax, N) like OASR
        # and OASP, so a non-equispaced source.frequencies vector would
        # be silently regridded by _resolve_freq_sweep inside the writer.
        # Warn (and resample) here so the user sees what is happening.
        source_freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
        if source_freqs.size > 1:
            _oases_resample_frequencies(source_freqs, self.model_name)

        # The OASN writer places a vertical array at x = y = 0;
        # receiver.ranges never reaches the deck.
        rcv_ranges = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
        if rcv_ranges.size > 1 or (rcv_ranges.size == 1 and rcv_ranges[0] > 0.0):
            warnings.warn(
                "OASN: receiver.ranges is ignored — OASN models a vertical "
                "array at x = y = 0 (depths only). Use the replica grid "
                "(xmin/xmax/...) for horizontal apertures.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        writer_kwargs = self._build_writer_kwargs()

        # The OASN options block must include the file output the run
        # mode produces. The user-supplied options (or default 'J') win
        # for additional letters (F, …); N (covariance) or R (replica)
        # is added automatically based on run_mode.
        user_options = self.options or 'J'
        opt_tokens = set(user_options.split())
        if run_mode == RunMode.COVARIANCE:
            opt_tokens.add('N')
        else:
            opt_tokens.add('R')
        # NOIPAR — and with it every noise argument below — runs only under
        # `IF (CALNSE.or.trfout)` (unoasn22.f:173-174), and CALNSE comes from
        # 'N'. A replica run that was given a noise field therefore needs 'N'
        # too, or the levels it was handed would never reach the deck.
        if self._has_noise_field():
            opt_tokens.add('N')
        elif run_mode == RunMode.COVARIANCE:
            # With no Block VI source the covariance is just the default
            # white-noise diagonal — 10**(-200/10) = 1e-20 per sensor with
            # zero cross terms (measured) — which reads like a computed
            # noise field but carries none.
            warnings.warn(
                "OASN: no noise source is configured, so the covariance "
                "matrix is only the white-noise floor — a diagonal of 1e-20 "
                "(the -200 dB default) with zero cross terms. Configure a "
                "noise field on the constructor: surface_noise_level=, "
                "deep_noise_level=, discrete_sources=, or an explicit "
                "white_noise_level=.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        options = ' '.join(sorted(opt_tokens))

        fm = self._setup_file_manager()
        try:
            base_name = 'oasn_run'
            input_file = fm.get_path(f'{base_name}.dat')
            self._log(f"Writing OASN input file: {input_file} (options={options})")
            write_oasn_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                options=options,
                **writer_kwargs,
            )
            proc = self._execute(base_name, fm.work_dir)

            if run_mode == RunMode.COVARIANCE:
                # FOR016 is always set (_oases_subprocess_env), so the
                # covariance can only appear under the .xsm name.
                cov_path = self._require_output(
                    [fm.get_path(f'{base_name}.xsm')],
                    what='a covariance file', process=proc,
                )
                self._log(f"Reading OASN covariance file: {cov_path}")
                cov_data = read_oasn_covariance(cov_path)

                rcv_pos = None
                rcv_depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
                if rcv_depths.size == cov_data['covariance'].shape[1]:
                    # OASES OASN config places receivers along the array;
                    # only depth varies in the typical uacpy usage.
                    rcv_pos = np.zeros((rcv_depths.size, 3), dtype=float)
                    rcv_pos[:, 2] = rcv_depths

                freqs = _oasn_freq_axis(cov_data)
                kw = self._result_kwargs(
                    source,
                    backend='oasn',
                    frequencies=freqs,
                    n_receivers=cov_data['n_receivers'],
                    title=cov_data.get('title', ''),
                )
                cov_result = Covariance(
                    covariance=cov_data['covariance'],
                    receiver_positions=rcv_pos,
                    **kw,
                )
                self._attach_output_paths(
                    cov_result, fm.work_dir, base_name,
                    primary_files=(('xsm_file', '.xsm'),),
                )
                return cov_result

            # RunMode.REPLICA — FOR014 is always set, so the replicas can
            # only appear under the .rpo name.
            rep_path = self._require_output(
                [fm.get_path(f'{base_name}.rpo')], what='a replica file', process=proc,
                hint=('Set the replica grid via OASN(xmin=…, xmax=…, nx=…, '
                      'zmin=…, zmax=…, nz=…) on the constructor.'),
            )
            self._log(f"Reading OASN replica file: {rep_path}")
            rep_data = read_oasn_replicas(rep_path)
            # x/y come back in km from the .rpo file; expose metres.
            replica_z = np.linspace(rep_data['z_min'], rep_data['z_max'], rep_data['n_z'])
            replica_x = np.linspace(rep_data['x_min'], rep_data['x_max'], rep_data['n_x']) * 1000.0
            replica_y = np.linspace(rep_data['y_min'], rep_data['y_max'], rep_data['n_y']) * 1000.0
            freqs = _oasn_freq_axis(rep_data)
            kw = self._result_kwargs(
                source,
                backend='oasn',
                frequencies=freqs,
                n_receivers=rep_data['n_receivers'],
                title=rep_data.get('title', ''),
            )
            rep_result = Replicas(
                replicas=rep_data['replicas'],
                replica_z=replica_z,
                replica_x=replica_x,
                replica_y=replica_y,
                receiver_positions=rep_data.get('receiver_positions'),
                **kw,
            )
            self._attach_output_paths(
                rep_result, fm.work_dir, base_name,
                primary_files=(('rpo_file', '.rpo'),),
            )
            return rep_result

        finally:
            fm.finish()

    # ``compute_covariance`` / ``compute_replicas`` come from the base class
    # (RunMode.COVARIANCE / RunMode.REPLICA dispatch).

    # Mirrors third_party/oases/bin/oasn: replica vectors on unit 14,
    # covariance on unit 16 (oasmun21_bin.f:335 reads FOR016 back), and the
    # unit-26 echo of the array and noise levels INPRCV opens at
    # oasnun22.f:37.
    _FOR_FILES = {
        'FOR014': 'rpo',
        'FOR016': 'xsm',
        'FOR026': 'chk',
    }
    # Every unit above is env-assigned, so the outputs always land on the
    # base_name suffixes (a replica run additionally leaves .021 behind).
    _OUTPUT_SUFFIXES = ('.xsm', '.rpo', '.plt', '.plp', '.chk', '.021')
    # A replica run dumps every replica vector as per-receiver ASCII to
    # unit 92 with a plain WRITE and no OPFILW — a debug block whose guard
    # is commented out (oasmun21_bin.f:157-162) — so a bare fort.92 stays
    # behind in the work dir.
    _OUTPUT_FORT_FILES = ('fort.92',)


class OASR(OASES):
    """
    OASR - OASES Reflection Coefficients Model

    Computes plane wave reflection coefficients at the bottom interface.

    Parameters
    ----------
    executable : Path, optional
        Path to OASR binary. Auto-detected if ``None``.
    angles : ndarray, optional
        Uniformly spaced angle grid (deg). Default
        ``linspace(0, 90, 181)``.
    angle_type : str, optional
        ``'grazing'`` (OASES native, default) | ``'incidence'``
        (converted via ``grazing = 90 - incidence``).
    reflection_type : str, optional
        ``'P-P'`` (effective default) | ``'P-SV'`` | ``'P-Slow'`` (Biot
        only) | ``'transmission'``. Translates to OASR option letter
        (``'N'`` / ``'S'`` / ``'B'`` / ``'t'``). Mutually exclusive with
        a raw ``options`` string.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Range-independent reflection-vs-angle/freq solver. Supports
    elastic / poro-elastic bottom layers. Returns reflection magnitude
    and phase per ``(frequency, angle)``. Pass ``freq_min``,
    ``freq_max``, ``n_frequencies`` on ``run()`` for a broadband sweep.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'bottom_range': 'median'`` (the layer stack is
    kept). SSP collapse left at the global ``'r0'`` because the
    SSP boundary speed is essentially irrelevant to the reflection
    coefficient.

    Examples
    --------
    >>> from uacpy.models import OASR
    >>> oasr = OASR(angles=np.linspace(0, 90, 100))
    >>> refl = oasr.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASR:
    # range-independent reflection vs angle/freq; layered bottom honoured.
    # Only the bottom stack matters — SSP collapse left at the global default
    # ('r0'), as the SSP is essentially irrelevant to the reflection coeff.
    # No 'rough_surface': the OASR deck has no sea surface — its layer 1 is
    # the water half-space the plane wave arrives through, and INENVI
    # discards that record's RG (oaseun31.f:377) — so surface roughness
    # cannot reach the computation and _project_environment discloses the
    # drop instead.
    spec = ModelSpec(
        modes=(RunMode.REFLECTION,),
        supports={'layered_bottom', 'elastic_media', 'rough_bottom'},
        # A plane-wave reflection coefficient is independent of source
        # geometry, and this deck writer proves it: OASR.run reads only
        # ``source.frequencies`` and ``oases_writer.py``'s OASR block never
        # touches ``source.depths`` or ``source.source_type``. Refusing a
        # source type nothing reads would break reusing one Source across
        # models, so every type is accepted — the same reasoning Bounce
        # states for the same run mode. This is the one place an OASES class
        # widens past ``('point',)``.
        source_types=VALID_SOURCE_TYPES,
        collapse={'bottom_range': 'median'},
    )
    source = 'oases'

    def __init__(
        self,
        executable: Optional[Path] = None,
        angles: Optional[np.ndarray] = None,
        angle_type: str = 'grazing',
        reflection_type: Optional[str] = None,
        options: Optional[str] = None,
        angle_output_increment: Optional[int] = None,
        freq_output_increment: Optional[int] = None,
        interface_roughness: Optional[list] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to OASR executable. Auto-detected if None.
        angles : ndarray, optional
            Angle grid in degrees, uniformly spaced. Default:
            ``np.linspace(0, 90, 181)``. OASR's deck carries only
            ``(ANGLE1, ANGLE2, NANG)`` and generates the grid itself
            (``unoasr21.f:173-177``), so a non-uniform array raises
            ``ConfigurationError`` rather than being silently resampled.
        angle_type : str, optional
            'grazing' (OASES native) or 'incidence' (converted via
            ``grazing = 90 - incidence`` before being written to the input
            file). Default: 'grazing'.
        reflection_type : str, optional
            One of 'P-P' (effective default), 'P-SV', 'P-Slow' (Biot
            only), or 'transmission'. Translates to the OASR option
            letter ('N' / 'S' / 'B' / 't').
        options : str, optional
            Raw OASES option string, written verbatim. ``None`` lets the
            wrapper derive it from ``reflection_type``. Passing both
            raises ``ConfigurationError`` — the deck would run whichever
            reflection the raw string selects, not the named one.
        angle_output_increment : int, optional
            OASR's ``NAOU``: decimates only OASES' own reflection-vs-angle
            *plot* curves. The ``.rco``/``.trc`` tables uacpy reads back
            carry every angle regardless, so this cannot change any number
            in the returned :class:`ReflectionCoefficient`. ``None`` leaves
            the knob out of the writer call, which applies its own
            ``max(1, n_angles // 10)`` — every 18th sample on the default
            181-angle grid, not every sample.
        freq_output_increment : int, optional
            OASR's ``NFOU`` (``unoasr21.f:116``): decimates only OASES' own
            reflection-vs-frequency *plot* curves; like ``NAOU`` it never
            touches the tables uacpy returns. ``None`` → the writer's
            ``max(1, n_frequencies // 10)``.
        interface_roughness : list, optional
            Per-interface RMS roughness in metres, one entry per layer
            record, ordered top → bottom. Entry 0 is OASR's layer-1
            (water half-space) record, whose RG INENVI discards
            (``oaseun31.f:377``); entry 1 is the reflecting water/seabed
            interface. An entry may be a float, or a ``(RG, CL, M)``
            tuple / ``{'RG', 'CL', 'M'}`` dict to select the Goff-Jordan
            roughness spectrum. Unset entries fall back to the
            environment's own interface roughness.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        self.angles = (
            np.asarray(angles, dtype=float) if angles is not None else None
        )
        self.angle_type = angle_type
        # Kept as passed so ``copy()`` round-trips and so an explicit value
        # alongside a raw ``options`` string is distinguishable from the
        # default. Resolved by ``_resolve_reflection_type``.
        self.reflection_type = reflection_type
        self.options = options
        if options is not None and reflection_type is not None:
            raise ConfigurationError(
                f"OASR: options={options!r} replaces the whole option line, "
                f"so reflection_type={reflection_type!r} would be recorded "
                f"in the result metadata without ever reaching the deck. "
                f"Pass either the raw string or reflection_type, not both."
            )
        self._reject_shear_reflection_types()
        self._reject_slowness_sampling()
        self.angle_output_increment = (
            int(angle_output_increment)
            if angle_output_increment is not None else None
        )
        self.freq_output_increment = (
            int(freq_output_increment)
            if freq_output_increment is not None else None
        )
        self.interface_roughness = (
            list(interface_roughness) if interface_roughness else None
        )

        # OASR is strictly a boundary-reflection solver; it does not produce
        # transmission loss. Declare that explicitly.
        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable, lambda: _oases_find_executable(self, 'oasr'),
        )

    def _oasr_is_log_swept(self) -> bool:
        """True when option ``'C'`` puts OASR on a logarithmic sweep.

        ``oasr.tex:135`` documents ``C`` as a plot option ("Loss contours
        plotted in frequency and grazing angle"), but ``unoasr21.f:123-125``
        computes ``F1LOG``/``DFLOG`` and ``:243`` evaluates the coefficients at
        ``EXP(F1LOG + (JJ-1)*DFLOG)`` — so it changes the frequencies the
        physics is computed at. ``oast.tex:200-203`` confirms it from the other
        side, calling ``C`` the way to obtain "consistent logarithmic sampling".
        """
        return 'C' in (self.options or '')

    def _reject_shear_reflection_types(self) -> None:
        """Refuse the two reflection types that are identically zero here.

        OASR reflects off the seabed, so the incident medium is the water
        column — a fluid, which carries no shear. `oasr.tex:143-145` defines
        option ``'S'`` as "the P-SV wave reflection coefficient"; with no SV
        wave possible in the incident medium that coefficient is zero for
        **every** environment uacpy can express, and OASES duly writes a
        column of zeros to the ``.rco``. Measured: `max|R| = 0.000000` for
        ``'P-SV'`` against `0.999947` for the default, with an elastic ice
        canopy on the surface making no difference — the canopy is not the
        incident medium at the seabed.

        ``'P-Slow'`` (option ``'B'``) is the Biot slow wave and needs a
        poro-elastic medium; no uacpy carrier expresses one.

        Raised rather than warned: there is no configuration in which either
        returns a usable number, so a warning would leave the caller holding
        an all-zero array that looks like a computed result.
        """
        rt = self._resolve_reflection_type()
        if rt not in ('P-SV', 'P-Slow'):
            return
        why = ("the incident medium at the seabed is the water column, a "
               "fluid, which carries no SV wave (oasr.tex:143-145)"
               if rt == 'P-SV' else
               "it is the Biot slow wave and needs a poro-elastic medium, "
               "which no uacpy carrier expresses")
        if self.reflection_type is not None:
            raise UnsupportedFeatureError(
                'OASR',
                f"reflection_type={rt!r} — {why}, so OASES returns a column "
                f"of zeros rather than a coefficient. Use "
                f"reflection_type='P-P' (the default) or 'transmission'",
            )
        # Reached via the raw ``options=`` escape hatch, which is documented
        # as written verbatim. Warn rather than raise so the override stays an
        # override — but the zeros are just as useless here.
        warnings.warn(
            f"OASR(options={self.options!r}) selects {rt}, which returns a "
            f"column of zeros: {why}. The deck is written as given.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )


    def _reject_slowness_sampling(self) -> None:
        """Reject raw ``options`` carrying ``'p'`` (slowness sampling).

        ``write_oasr_input`` always writes Block V as grazing angles in
        degrees; under ``'p'`` the binary reads those same three numbers as
        slownesses in s/km (oasjun21.f:36-37), so the computed grid would
        not be the one the deck asked for.
        """
        if 'p' not in (self.options or ''):
            return
        raise UnsupportedFeatureError(
            'OASR',
            "option 'p' (slowness sampling): write_oasr_input writes "
            "Block V in grazing degrees, which oasjun21.f:36-37 would "
            "reinterpret as slownesses in s/km — the computed grid would "
            "not be the one requested",
            alternatives=["angle sampling (the default)",
                          "angle_type='incidence'"],
            alternatives_label='samplings',
        )

    def _resolve_reflection_type(self) -> str:
        """The coefficient the deck actually computes.

        Derived from the raw ``options`` letters when one is pinned, so
        the recorded provenance describes the run rather than an unused
        constructor argument. Falls back to ``'P-P'``, OASES' own default
        when no reflection letter appears.

        ``REFL_TYPE_TO_OPTION`` lists the four as peers, but GETOPT does not
        treat them that way (``unoasr21.f:349-378``): ``N``/``S``/``B`` each
        assign ``IPARM`` — the *wave parameter* — while ``'t'`` flips the
        independent ``transmit`` flag, and ``oasjun21.f:80-84`` then picks
        ``trcoef(iparm)`` over ``rfcoef(iparm)``. So ``'N T t'`` is a P-wave
        *transmission* run, not a contradiction: ``'t'`` decides which of the
        two tables is read and wins here, and among the parameter letters the
        last one in the line wins, because each occurrence overwrites
        ``IPARM``.
        """
        if self.options is None:
            return self.reflection_type or 'P-P'
        from uacpy.io.oases_writer import REFL_TYPE_TO_OPTION
        opts = str(self.options)
        if REFL_TYPE_TO_OPTION['transmission'] in opts:
            return 'transmission'
        by_letter = {letter: name
                     for name, letter in REFL_TYPE_TO_OPTION.items()
                     if name != 'transmission'}
        seen = [by_letter[ch] for ch in opts if ch in by_letter]
        return seen[-1] if seen else 'P-P'

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """Run OASR.

        Returns a :class:`ReflectionCoefficient` with ``theta`` (1-D angle
        grid in degrees) and ``R``/``phi`` shaped ``(n_angles, n_frequencies)``
        when more than one frequency was requested. Single-frequency runs
        return 1-D ``R``/``phi``.

        OASES uses grazing angles natively. When ``angle_type='incidence'``,
        input angles are converted via ``grazing = 90 - incidence`` before
        being written to the OASR input file.

        Parameters
        ----------
        frequencies : ndarray, optional
            Explicit frequency vector (Hz). When provided, overrides
            ``source.frequencies``. OASES treats the sweep as
            equispaced — uacpy resamples and warns if your vector is not.
        source_waveform, sample_rate, output_duration
            Accepted for the polymorphic ``run()`` contract; OASR computes
            reflection coefficients only, so a non-``None`` value warns.
        """
        self._require_run_triple(env, source, receiver)
        run_mode = self._resolve_run_mode(run_mode)
        self._warn_ignored_run_kwargs(
            run_mode,
            reason='OASR computes plane-wave reflection coefficients only',
            source_waveform=source_waveform,
            sample_rate=sample_rate,
            output_duration=output_duration,
        )

        # Re-validate in case the caller mutated ``options`` after __init__;
        # the .trc-first output search below relies on 'p' being absent.
        self._reject_slowness_sampling()

        angles = self.angles if self.angles is not None else np.linspace(0, 90, 181)

        writer_kwargs: dict = {}
        if frequencies is not None:
            freqs_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
            if freqs_arr.size == 0:
                raise ConfigurationError(
                    "OASR.run(frequencies=…) requires at least one "
                    "positive frequency."
                )
            fmin, fmax, n_freq, _ = _oases_resample_frequencies(
                freqs_arr, 'OASR', log_spaced=self._oasr_is_log_swept(),
            )
            writer_kwargs['freq_min'] = fmin
            writer_kwargs['freq_max'] = fmax
            writer_kwargs['n_frequencies'] = n_freq
        else:
            # OASR expresses the frequency axis as (fmin, fmax, N), so a
            # non-equispaced source.frequencies vector would be silently
            # regridded by the writer. Warn (and resample) here so the user
            # sees it — mirrors OASN/OASP.
            source_freqs = np.atleast_1d(
                np.asarray(source.frequencies, dtype=float))
            if source_freqs.size > 1:
                _oases_resample_frequencies(
                    source_freqs, 'OASR',
                    log_spaced=self._oasr_is_log_swept())

        if self.options is not None:
            writer_kwargs['options'] = self.options
        if self.angle_output_increment is not None:
            writer_kwargs['angle_output_increment'] = self.angle_output_increment
        if self.freq_output_increment is not None:
            writer_kwargs['freq_output_increment'] = self.freq_output_increment
        if self.interface_roughness is not None:
            writer_kwargs['interface_roughness'] = self.interface_roughness

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        fm = self._setup_file_manager()

        try:
            base_name = 'oasr_run'
            input_file = fm.get_path(f'{base_name}.dat')

            self._log(f"Writing OASR input file: {input_file}")
            write_oasr_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                angles=angles,
                angle_type=self.angle_type,
                reflection_type=self._resolve_reflection_type(),
                **writer_kwargs,
            )

            proc = self._execute(base_name, fm.work_dir)

            # Under option 'T' REFLEC writes both tables on every sample —
            # slowness in s/km to unit 22 (.rco) and grazing angle in degrees
            # to unit 23 (.trc), oasjun21.f:103-104 — so the pair holds the
            # same coefficients under two abscissae and the choice here is
            # uacpy's. Grazing degrees is always the right one: the only
            # option that would make the slowness abscissa the requested grid
            # is 'p' (oasjun21.f:36-37 rereads ANGLE1/ANGLE2/NANG as
            # slowness), and _reject_slowness_sampling refuses that outright,
            # in __init__ and again above.
            search = ['.trc', '.rco']

            output_file = self._require_output(
                [fm.get_path(f'{base_name}{ext}') for ext in search],
                what='a reflection-coefficient table', process=proc,
            )

            self._log(f"Reading OASR output: {output_file}")
            data = read_oasr_reflection_coefficients(output_file)

            theta_arr, R_arr, phi_arr, freqs_arr = _stack_oasr_data(data)
            field = ReflectionCoefficient(
                theta=theta_arr, R=R_arr, phi=phi_arr,
                **self._result_kwargs(
                    source,
                    backend='oasr',
                    frequencies=freqs_arr if len(freqs_arr) else None,
                    sampling_type=data.get('sampling_type', 'angle'),
                    reflection_type=self._resolve_reflection_type(),
                ),
            )
            self._attach_output_paths(
                field, fm.work_dir, base_name,
                primary_files=(
                    ('trc_file', '.trc'),
                    ('rco_file', '.rco'),
                ),
            )

            self._log("OASR simulation complete")
            return field

        finally:
            fm.finish()

    # Units 22 and 23 are the two reflection-coefficient tables option 'T'
    # opens (unoasr21.f:201-205). They are written together, not either/or:
    # oasjun21.f:103 puts slowness on 22 and :104 grazing angle on 23, which
    # is why both files can exist and run() picks by the 'p' option.
    # third_party/oases/bin/oasr also names FOR002 and FOR004, but those
    # units OASR never opens, so they need no entries here.
    _FOR_FILES = {
        'FOR022': 'rco',
        'FOR023': 'trc',
    }
    # Every unit is env-assigned, so the outputs always land on the
    # base_name suffixes; a run additionally leaves .021 behind.
    # '.045' as for OAST: option 's' (SCTOUT, unoasr21.f:400) dumps the
    # boundary operators there, and a stale one would be read as this run's.
    _OUTPUT_SUFFIXES = ('.rco', '.trc', '.plt', '.plp', '.021', '.045')
    _OUTPUT_FORT_FILES = _SCTOUT_BARE_FORT46


class OASP(OASES):
    """
    OASP - OASES Pulse / Broadband Transfer-Function Model

    Computes broadband acoustic transfer functions via wavenumber integration
    followed by FFT to produce time-series / pulse responses. OASP is the
    "Pulse" variant of SAFARI, using the same wavenumber-integration kernel
    as OAST but evaluated across a frequency sweep.

    Parameters
    ----------
    executable : Path, optional
        Path to OASP binary. Auto-detected if ``None``.
    n_time_samples : int, optional
        Number of FFT time samples. Default ``4096``.
    freq_min : float, optional
        Lower edge of the FFT frequency sweep (Hz). Default ``0.0``.
    freq_max : float, optional
        Upper edge of the FFT frequency sweep (Hz). ``None`` (default)
        derives ``2.5 × `` the centre frequency at ``run()`` time.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Everything after ``n_time_samples`` is keyword-only — see
    :meth:`__init__` for why this model alone is written that way.

    Notes
    -----
    Broadband-oriented; supports ``RunMode.BROADBAND`` (returns a
    :class:`Field`) and ``RunMode.TIME_SERIES`` (returns a
    :class:`Field` after ``synthesize_time_series``). For a
    single-cell trace use ``tf.to_time_trace(depth, range)``. Can be
    expensive (reduce ``n_time_samples`` for speed). For range-dependent
    problems, RAM is recommended.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom_range': 'median'`` (the
    layer stack is kept).

    Examples
    --------
    >>> from uacpy.models import OASP
    >>> oasp = OASP(n_time_samples=256, freq_max=120)
    >>> result = oasp.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASP:
    # range-independent broadband wavenumber integration / pulse synthesis;
    # multi-layer fluid + elastic bottom honoured. Single spectral solve per
    # frequency → mean SSP / median bottom column represent the path.
    spec = ModelSpec(
        modes=(RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES),
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
    )
    source = 'oases'

    def __init__(
        self,
        executable: Optional[Path] = None,
        n_time_samples: int = 4096,
        *,
        freq_min: float = 0.0,
        freq_max: Optional[float] = None,
        center_frequency: Optional[float] = None,
        freq_output_increment: Optional[int] = None,
        options: Optional[str] = None,
        range_start: Optional[float] = None,
        integration_offset: float = 0.0,
        nw_samples: int = -1,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        dip_angle: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        """
        Every parameter after ``n_time_samples`` is keyword-only, which no
        sibling OASES model does. The reason is specific to this one: OASP
        shipped with its band edges declared ``freq_max, freq_min``, the
        reverse of everything downstream of them. Block VIII is
        ``NX FR1 FR2 DT ...`` (``unoasp22.f:176``) with FR1 the LOW index
        (``:238-239`` compute ``LX = FR1/DLFREQ + 1``, ``MX = FR2/DLFREQ + 2``),
        :func:`~uacpy.io.oases_writer.write_oasp_input` writes the pair
        min-first, and :class:`OASSP` declares it min-first. Correcting the
        order alone would have rebound every positional ``OASP(exe, n, a, b)``
        to the opposite physics with no error raised — and an inverted pair
        means ``LX > MX``, so the kernel's frequency loop runs zero times.
        The bare ``*`` makes those calls a ``TypeError`` instead. Treat this
        as local damage control, not a house style.

        Parameters
        ----------
        executable : Path, optional
            Path to OASP executable. Auto-detected if None.
        n_time_samples : int, optional
            Power-of-two FFT length (samples per receiver trace).
            Default 4096.
        freq_min : float, optional
            Lower edge of the OASP broadband sweep (Hz). Default 0.0.
            Must be below ``freq_max`` when that is pinned.
        freq_max : float, optional
            Upper edge of the OASP broadband sweep (Hz). ``None``
            (default) derives ``2.5 ×`` the centre frequency at
            ``run()`` time; a pinned value below the centre frequency
            or the top of the requested band raises at ``run()``.
        center_frequency : float, optional
            Carrier frequency for the pulse (Hz). ``None`` defaults to
            the centre (midpoint) of the run's frequency band at
            ``run()`` time — a single ``source.frequencies`` entry is
            its own centre.
        freq_output_increment : int, optional
            OASP's ``INTF`` (Block VII, ``unoasp22.f:160``). It gates how
            often the wavenumber *integrand* is plotted
            (``unoasp22.f:591``: ``IF (MOD(JJ-LXP1,INTF).EQ.0) KPLOT=1``,
            consumed at ``unoasp22.f:678``) and does **not** decimate the
            ``.trf`` frequency axis, which always carries every bin
            ``LXP1..MX``. ``None`` → 40.
        options : str, optional
            Raw OASES option string. ``None`` defers to the writer's
            default (``'N J'``); the ``'J'`` forces a real frequency axis
            (OMEGIM=0) so the broadband ``.trf`` synthesis stays valid.
        range_start : float, optional
            First receiver range (m). ``None`` defaults to
            ``receiver.ranges.min()``.
        integration_offset : float, optional
            Wavenumber-contour offset (dB/wavelength). Default 0.
        nw_samples : int, optional
            Wavenumber sample count. ``-1`` lets OASP auto-pick.
        c_low, c_high : float, optional
            ``CMIN``/``CMAX`` (m/s), Block VII's phase-speed window.
            ``None`` derives the pair from the water column alone, which
            leaves an elastic seabed's shear and interface branches — phase
            speeds below the slowest water speed — outside ``k_max`` and out
            of the integral. Same knob and same reason as :class:`OAST`.
        dip_angle : float, optional
            Fault dip angle (degrees) for the dip-slip moment source that
            option ``'4'`` selects (``unoasp22.f:993-995``). INSRC reads it
            from the source record — as the second token without ``'L'``
            (``oaseun31.f:1114``), as the seventh with it
            (``oaseun31.f:1089``). ``None`` writes 0 when ``'4'`` is
            present and raises when it is not.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        self.n_time_samples = int(n_time_samples)
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max) if freq_max is not None else None
        # An inverted pair reaches the deck unvalidated (the writer formats
        # whatever it is given) and OASP then computes LX > MX, so the
        # frequency loop never executes and the .trf comes back empty rather
        # than wrong. run()'s own check compares freq_max against the band
        # top only, never against freq_min, so this is the one place the
        # ordering is tested.
        if self.freq_max is not None and self.freq_min >= self.freq_max:
            raise ConfigurationError(
                f"OASP: freq_min={self.freq_min:g} Hz is not below "
                f"freq_max={self.freq_max:g} Hz. Block VIII writes the two as "
                f"FR1 FR2 (unoasp22.f:176) and the kernel indexes them as "
                f"LX = FR1/DLFREQ + 1 .. MX = FR2/DLFREQ + 2 "
                f"(unoasp22.f:238-239), so an inverted pair gives LX > MX and "
                f"the frequency loop runs zero times.",
                remediation="Pass the low edge as freq_min and the high edge "
                            "as freq_max.",
            )
        self.center_frequency = (
            float(center_frequency) if center_frequency is not None else None
        )
        self.freq_output_increment = (
            int(freq_output_increment)
            if freq_output_increment is not None else None
        )
        self.options = options
        self.range_start = (
            float(range_start) if range_start is not None else None
        )
        self.integration_offset = float(integration_offset)
        self.nw_samples = int(nw_samples)
        # Block VII's CMIN/CMAX; ``None`` keeps the writer's water-column
        # derivation, as on OAST.
        self.c_low = float(c_low) if c_low is not None else None
        self.c_high = float(c_high) if c_high is not None else None
        # OASP reads the offset token under 'J' or 'd' (unoasp22.f:126-133)
        # and otherwise sets OFFDB = OFFDBIN = 0 at :365. ``options=None``
        # takes the writer's default line.
        _warn_offset_ignored_under_auto_sampling(
            'OASP', self.integration_offset, self.nw_samples,
            options=self.options if self.options is not None else 'N J',
            offset_letters='Jd')
        # DA, the dip angle INSRC reads only under the '4' (dip-slip) option.
        self.dip_angle = float(dip_angle) if dip_angle is not None else None

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable, lambda: _oases_find_executable(self, 'oasp'),
        )

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """
        Run OASP broadband wavenumber-integration computation.

        Parameters
        ----------
        env : Environment
            Ocean environment (range-independent; range-dependent
            features are collapsed with a warning)
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        run_mode : RunMode, optional
            ``COHERENT_TL`` (default) — extract TL at source frequency.
            ``BROADBAND`` — full H(f).
            ``TIME_SERIES`` — real pressure p(t); requires
            ``source_waveform`` + ``sample_rate``.
        source_waveform : ndarray, optional
            Source pulse for ``TIME_SERIES`` mode.
        sample_rate : float, optional
            Sampling rate of ``source_waveform`` in Hz.
        output_duration : float, optional
            Desired output duration (seconds) for ``TIME_SERIES``. When
            given, the source waveform is zero-padded internally so the
            broadband frequency grid is tight enough (``Δf =
            1/output_duration``) to span this window without DFT
            wraparound. Defaults to ``len(source_waveform)/sample_rate``.

        Returns
        -------
        result : Result
            :class:`Field` for COHERENT_TL, :class:`Field`
            for BROADBAND, :class:`Field` for TIME_SERIES.
        """
        self._require_run_triple(env, source, receiver)
        n_time_samples = self.n_time_samples
        freq_max = self.freq_max
        freq_min = self.freq_min

        run_mode = self._resolve_run_mode(run_mode)
        if run_mode not in (RunMode.BROADBAND, RunMode.TIME_SERIES):
            # BROADBAND is covered inside ``_prepare_timeseries``; warning
            # here too would double up.
            self._warn_ignored_run_kwargs(
                run_mode,
                source_waveform=source_waveform,
                sample_rate=sample_rate,
                output_duration=output_duration,
            )
        source_waveform, frequencies = self._prepare_timeseries(
            run_mode, source, frequencies, source_waveform, sample_rate,
            output_duration,
        )

        # COHERENT_TL is a single-frequency mode (validate_inputs rejects a
        # multi-frequency Source for it). An explicit multi-element
        # ``frequencies=`` would otherwise bypass that guard — OASP would run a
        # full broadband sweep and silently return only the bin nearest the
        # source frequency. Reject it the same way, pointing at BROADBAND.
        if run_mode == RunMode.COHERENT_TL and frequencies is not None:
            if np.atleast_1d(np.asarray(frequencies)).size > 1:
                raise ConfigurationError(
                    "OASP.run(run_mode=COHERENT_TL) takes a single frequency; "
                    "got a multi-element frequencies= vector. For broadband "
                    "H(f) use RunMode.BROADBAND."
                )

        self._reject_unreadable_options()

        if frequencies is not None:
            freqs_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
            if freqs_arr.size == 0:
                raise ConfigurationError(
                    "OASP.run(frequencies=…) requires at least one positive "
                    "frequency."
                )
            # OASP rebuilds its own (n_time_samples, freq_max) grid; an
            # explicit ``frequencies=`` vector overrides ``freq_max`` to the
            # user's max and ``n_time_samples`` so the resulting OASP bins
            # span at least the requested band. The (fmin, fmax, N) triple
            # is OASP's internal language — warn if the user vector is
            # non-equispaced so they know the resulting bins won't match
            # their input one-to-one.
            fmin_user, freq_max, n_user, _ = _oases_resample_frequencies(
                freqs_arr, 'OASP',
            )
            # Deck fc: the centre of the requested band — the same centre
            # convention the sibling broadband models use (RAM's (fc, Q, T)
            # sweep is parameterised around the band midpoint). A pinned
            # ``center_frequency`` wins.
            fc_run = (
                self.center_frequency
                if self.center_frequency is not None
                else float(0.5 * (freqs_arr.min() + freqs_arr.max()))
            )
            if self.freq_min and abs(self.freq_min - fmin_user) > 1e-9:
                warnings.warn(
                    f"OASP.run(frequencies=…) sets the sweep's lower edge to "
                    f"{fmin_user:.3f} Hz, overriding the constructor's "
                    f"freq_min={self.freq_min:.3f} Hz.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
            if (self.freq_max is not None
                    and abs(self.freq_max - freq_max) > 1e-9):
                warnings.warn(
                    f"OASP.run(frequencies=…) sets the sweep's upper edge to "
                    f"{freq_max:.3f} Hz, overriding the constructor's "
                    f"freq_max={self.freq_max:.3f} Hz.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
            freq_min = fmin_user
            if n_user > 1:
                # df implied by the equispaced (fmin, fmax, N) grid that
                # OASES will actually run.
                df_user = (freq_max - fmin_user) / (n_user - 1)
            else:
                df_user = float(freqs_arr[0])
            if df_user > 0:
                # OASP's bin spacing is DLFREQ = 1/(DT*NX) (unoasp22.f:237)
                # and the writer sets DT at Nyquist, 1/(2*freq_max), so
                # NX = 2*freq_max/DLFREQ samples are needed to land bins as
                # fine as the user's df. OASP requires NT = 2^M
                # (oasp.tex:129), so round that up to a power of two — never
                # down past whatever n_time_samples already asked for.
                target = max(int(n_time_samples or 0),
                             int(np.ceil(2.0 * freq_max / df_user)))
                if target > 1:
                    n_time_samples = 1 << (target - 1).bit_length()
                else:
                    n_time_samples = 2
        else:
            # Centre convention shared with the sibling broadband models
            # (RAM's (fc, Q, T) sweep centres on the band midpoint): a
            # multi-element ``source.frequencies`` names a band, and its
            # centre — not its lower edge — sets the deck fc and the derived
            # sweep top. ``frequencies[0]`` as the centre put the top of the
            # band above the 2.5*fc sweep edge, never computed.
            src_freqs = np.atleast_1d(
                np.asarray(source.frequencies, dtype=float))
            fc_run = (
                self.center_frequency
                if self.center_frequency is not None
                else float(0.5 * (src_freqs.min() + src_freqs.max()))
            )
            f_top = max(fc_run, float(src_freqs.max()))
            derived_freq_max = freq_max is None
            if derived_freq_max:
                # Band headroom above the carrier, so the pulse's upper
                # sidelobes fall inside FR2 rather than being truncated.
                # OASES ties FR2 to nothing but FR1 (oasp.tex:131), so the
                # 2.5 is uacpy's default, not a model constraint.
                freq_max = 2.5 * fc_run
            # The derived edge needs the same test as a pinned one. It is
            # 2.5x the band centre, so it clears the band by construction —
            # unless ``center_frequency`` pins fc below it, and then a
            # 1000 Hz source with center_frequency=10 writes FR2=25 Hz and
            # comes back labelled with a 25 Hz bin.
            if f_top > freq_max:
                origin = (f"the derived sweep edge freq_max=2.5×fc="
                          f"{freq_max:.1f} Hz" if derived_freq_max
                          else f"the pinned sweep edge "
                               f"freq_max={freq_max:.1f} Hz")
                fix = ("Raise center_frequency to the band's own centre (or "
                       "leave it None), pin freq_max above the band, or pass "
                       "frequencies= explicitly."
                       if derived_freq_max else
                       "Raise freq_max (or leave it None to derive 2.5×fc), "
                       "or pass frequencies= explicitly.")
                raise ConfigurationError(
                    f"OASP: the requested band reaches {f_top:.1f} Hz "
                    f"(centre {fc_run:.1f} Hz), above {origin} — the sweep "
                    f"would never compute the top of the band. {fix}"
                )

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        fm = self._setup_file_manager()

        writer_kwargs: dict = {
            'integration_offset': self.integration_offset,
            'nw_samples': self.nw_samples,
            'freq_min': freq_min,
            # Both branches above resolve fc_run (pinned value or band
            # centre); the writer would otherwise default the deck fc to
            # source.frequencies[0].
            'center_frequency': fc_run,
        }
        if self.freq_output_increment is not None:
            writer_kwargs['freq_output_increment'] = self.freq_output_increment
        if self.options is not None:
            writer_kwargs['options'] = self.options
        if self.range_start is not None:
            writer_kwargs['range_start'] = self.range_start
        if self.c_low is not None:
            writer_kwargs['c_low'] = self.c_low
        if self.c_high is not None:
            writer_kwargs['c_high'] = self.c_high
        if self.dip_angle is not None:
            writer_kwargs['dip_angle'] = self.dip_angle

        try:
            base_name = 'oasp_run'
            input_file = fm.get_path(f'{base_name}.dat')
            self._log(f"Writing OASP input file: {input_file}")
            write_oasp_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                n_time_samples=n_time_samples,
                freq_max=freq_max,
                **writer_kwargs,
            )

            # Run executable
            proc = self._execute(base_name, fm.work_dir)
            self._reject_wavenumber_overrun(proc)
            # Option '8' renames the transfer function to <base>.dtrf
            # (oasiun23.f:837-845 `bufch='d'//trfext`); the record layout is
            # unchanged, CFFX being declared plain COMPLEX (oasiun23.f:18).
            output_file = self._require_output(
                [fm.get_path(f'{base_name}.trf'),
                 fm.get_path(f'{base_name}.dtrf'),
                 fm.get_path(f'{base_name}.plt')], what='a transfer-function file',
                process=proc,
            )
            self._log(f"Reading OASP output: {output_file}")
            trf_data = read_oasp_trf(output_file,
                                     receiver_depths=receiver.depths)

            return self._assemble_oasp_result(
                trf_data, env, source, fm, base_name,
                run_mode=run_mode, frequencies=frequencies,
                n_time_samples=n_time_samples, freq_max=freq_max,
                source_waveform=source_waveform, sample_rate=sample_rate)

        finally:
            fm.finish()

    def _assemble_oasp_result(self, trf_data, env, source, fm, base_name, *,
                              run_mode, frequencies, n_time_samples, freq_max,
                              source_waveform, sample_rate):
        """Turn a finished OASP ``.trf`` read into the :class:`Field` the run
        mode asks for: the whole complex H(f) cube for BROADBAND, one
        frequency slice for COHERENT_TL, or the synthesised trace for
        TIME_SERIES.

        Split from :meth:`OASP.run` so the deck-write / launch / read half of
        the run does not share a scope with the assembly; nothing here touches
        the binary or the work dir except to stamp the output paths.
        """
        transfer_func = trf_data['transfer_function']  # shape: (n_frequencies, n_range, n_depth)

        if run_mode in (RunMode.BROADBAND, RunMode.TIME_SERIES):
            # A multi-element Source.frequencies names bins just as
            # explicitly as run(frequencies=) does.
            asked = frequencies
            if asked is None and np.atleast_1d(
                    np.asarray(source.frequencies)).size > 1:
                asked = source.frequencies
            _warn_if_trf_grid_replaced_request(asked, trf_data['freq'])
            # Convention: (n_depth, n_range, n_frequencies) — trailing
            # axis is the variable dim. Source axes: (freq, range, depth).
            # The raw .trf payload is the travelling-wave field times -1:
            # measured against Scooter's Hankel path on the same Pekeris
            # case, arg(raw/Scooter) = 176.8-179.9 deg with the magnitude
            # ratio 0.94-1.02, so the negation is what makes the
            # 'travelling_wave' tag below mean the same convention every
            # coherent engine shares (field.exe gets the identical
            # treatment in kraken.py's _assemble_field_from_shd).
            tf_reordered = -np.transpose(transfer_func, (2, 1, 0))

            result = Field(
                data=tf_reordered,
                coords={
                    'depth': trf_data['depths'],
                    'range': trf_data['ranges'],
                    'frequency': trf_data['freq'],
                },
                **self._result_kwargs(
                    source,
                    phase_reference='travelling_wave',
                    backend='oasp',
                    frequencies=trf_data['freq'],
                    source_depth=trf_data['source_depth'],
                    center_frequency=trf_data['center_frequency'],
                    n_time_samples=n_time_samples,
                    freq_max=freq_max,
                ),
            )
            # Physical fastest compressional speed in the waveguide
            # (water column + sediment + half-space): the time-series
            # synthesis helpers anchor their output window at r / c_max,
            # ahead of the earliest bottom-refracted arrival.
            c_max = self._resolve_c_max(env)
            if c_max is not None:
                result.metadata['c_max'] = c_max
            result = _mask_zero_range(result, source, 'OASP')
            if run_mode == RunMode.TIME_SERIES:
                result = result.synthesize_time_series(
                    source_waveform=source_waveform,
                    sample_rate=sample_rate,
                )
        else:
            # COHERENT_TL: pick the bin nearest the requested frequency
            # and return the complex narrowband pressure (transposed
            # to the (n_depth, n_range) layout). Users get TL via
            # ``field.db`` or ``.to_db()``.
            f_req = (float(np.atleast_1d(frequencies)[0])
                     if frequencies is not None
                     else float(source.frequencies[0]))
            freq_idx = 0
            if len(trf_data['freq']) > 1:
                freq_diff = np.abs(trf_data['freq'] - f_req)
                freq_idx = np.argmin(freq_diff)
            # The ladder rarely lands a bin exactly on the request
            # (measured: 100 Hz asked, 99.97558 Hz returned, a phase
            # error growing to ~36 deg by 6 km), so the substitution is
            # announced just as the BROADBAND branch announces its axis.
            _warn_if_trf_grid_replaced_request(
                f_req, trf_data['freq'][freq_idx])

            # One frequency slice of the same .trf array the BROADBAND
            # branch returns, negated for the same reason: the raw
            # payload is the travelling-wave field times -1.
            p_at_freq = -transfer_func[freq_idx, :, :].T.astype(
                np.complex128)  # (n_d, n_r)

            result = Field(
                data=p_at_freq,
                coords={
                    'depth': trf_data['depths'],
                    'range': trf_data['ranges'],
                },
                **self._result_kwargs(
                    source,
                    phase_reference='travelling_wave',
                    backend='oasp',
                    frequencies=float(trf_data['freq'][freq_idx]),
                    frequencies_available=trf_data['freq'],
                    source_depth=trf_data['source_depth'],
                    center_frequency=trf_data['center_frequency'],
                    n_time_samples=n_time_samples,
                    freq_max=freq_max,
                ),
            )
            result = _mask_zero_range(result, source, 'OASP')

        self._attach_output_paths(
            result, fm.work_dir, base_name,
            primary_files=(('trf_file', '.trf'),),
        )

        self._log("OASP simulation complete")
        return result


    def _reject_unreadable_options(self) -> None:
        """Reject raw ``options`` letters whose ``.trf`` uacpy cannot read back.

        Every case below writes a well-formed ``.trf`` that the reader parses
        without complaint but that no longer means what the axis labels say —
        an extra output component the reader flattens away, a range axis that
        is really slowness, or a spectrum carrying a complex-contour offset
        the time-series synthesis does not undo. Left unchecked they surface
        as a plausible wrong answer, so each raises here instead. Leaving
        ``options`` unset (the writer's default ``'N J'``) hits none of them
        and skips the whole check.

        OASES GETOPT (``unoasp22.f:871-872`` ``READ(1,200) OPT`` /
        ``200 FORMAT(40A1)``, scanned character by character at ``:873``)
        ignores whitespace, so ``'NJO'`` enables ``'O'`` exactly like
        ``'N J O'`` — hence the tests below are on the character set, not on
        whitespace-split tokens.
        """
        if not self.options:
            return
        opt_chars = set(str(self.options)) - set(' \t\n')

        # The .trf reader collapses the MSUFT / ISROW / NOUT axes onto the
        # first slot, so extra output components are dropped, not reported.
        # Letters that add an IOUT slot, i.e. increment NOUT and put a second
        # component in every .trf record (unoasp22.f:887-918: N→1 V→2 H→3 R→5
        # K→6 S→7), plus 'U' (DECOMP), which splits the output over NCOMPO=5
        # separate files (unoasp22.f:455-456, opened one unit apart at
        # :539-540).
        multi_axis = {'V', 'H', 'R', 'K', 'S', 'U'} & opt_chars
        if multi_axis:
            raise ConfigurationError(
                f"OASP.run: options {sorted(multi_axis)} request "
                "multi-component / decomposed output, which the .trf "
                "reader currently flattens. Pass options without "
                "these letters (default 'N J' returns scalar pressure) "
                "or read the .trf directly."
            )
        if 'O' in opt_chars:
            # 'O' moves the frequency integration onto a complex
            # contour (Im(omega) = -ln(50)·Δf, unoasp22.f:372-373). The
            # .trf reader discards that offset and synthesize_time_series
            # does not re-apply exp(-Im(omega)·t), so the time series
            # would be silently wrong. Reject rather than mis-synthesise.
            raise ConfigurationError(
                "OASP.run: option 'O' (complex frequency integration) "
                "bakes an exp(-ln(50)·Δf·t) contour into the spectrum "
                "that uacpy's time-series synthesis does not undo. "
                "Drop 'O' (the default 'N J' uses a real frequency axis)."
            )
        if 't' in opt_chars:
            # Lowercase 't' sets INTTYP=-1 (unoasp22.f:1028-1029), and
            # unoasp22.f:178-189 then overwrites the deck's R0 / RSPACE /
            # NPLOTS with a slowness axis derived from CMIN/CMAX. The units are
            # s/km, not s/m: ``r0 = 1e3/cmaxin`` and
            # ``rspace = (1e3/cminin - r0)/(nwvno-1)`` at :186-188 carry the
            # 1e3. The amplitudes differ too — the tau-p branch writes
            # ``1.0E6*real(CFFX(...))`` (oasiun23.f:452-453) where the two
            # normal paths write the bare value (:318, :770), so a .trf read
            # without dividing it out is 120 dB high. uacpy would label the
            # abscissa 'range' in metres and the amplitude unscaled.
            raise ConfigurationError(
                "OASP.run: option 't' (tau-p seismograms) replaces the "
                "receiver range axis with slowness "
                "(unoasp22.f:178-189), which uacpy's Field has no "
                "coordinate for and would mislabel as range. Drop 't'."
            )
        if 'J' not in opt_chars and self.nw_samples < 1:
            # Under automatic wavenumber sampling (nw_samples < 1 →
            # AUSAMP), OASES forces the complex frequency contour
            # OMEGIM = -ln(50)·Δf unless 'J' keeps ICNTIN > 0
            # (unoasp22.f:288-296, :304-306 then :372-373). Without 'J'
            # the .trf then carries the same offset 'O' would, which the
            # time-series synthesis cannot undo.
            raise ConfigurationError(
                "OASP.run: a custom options string without 'J' enables the "
                "complex frequency contour (OMEGIM≠0) under automatic "
                "wavenumber sampling, which uacpy's time-series synthesis "
                "cannot undo. Add 'J' (the default 'N J' keeps a real "
                "frequency axis) or pin nw_samples≥1."
            )

    _FOR_FILES = {
        'FOR002': 'src',
    }
    # '.045'/'.046' are the option-'s' kernel files (SCTOUT, unoasp22.f:1039;
    # unit 46 opened at :253). They are OASSP *inputs* — _run_mean_field
    # requires both from this run's stem — so a pinned work_dir holding a
    # previous sweep's pair would hand oassp2 the wrong kernels, and they are
    # megabytes each.
    _OUTPUT_SUFFIXES = ('.trf', '.dtrf', '.plt', '.plp', '.045', '.046')
    _OUTPUT_FORT_FILES = _SCTOUT_BARE_FORT46


class OASSP(OASES):
    """
    OASSP — OASES Scattered-Field Realization Model

    Generates one **realization** of the broadband field scattered from a
    rough interface, from the mean-field boundary operators an OASP run writes
    with option ``'s'``. OASSP is a post-processor, so ``run()`` drives the
    mean-field binary first and then ``oassp2``; the mean-field
    :class:`~uacpy.core.results.Field` is kept on
    ``metadata['mean_field_result']`` so the coherent field is not thrown away.

    Where :class:`OASP` returns the coherent field, OASSP returns the
    scattered one — same ``.trf`` format, same reader, same grid.

    Parameters
    ----------
    executable : Path, optional
        Path to the ``oassp2`` binary. Auto-detected if ``None``.
    correlation_length : float
        ``CL`` (m) of the interface roughness power spectrum. Required:
        ``oass.tex:182-183`` — the roughness statistics "must be specified.
        These are not adopted from OASR or OAST".
    spectral_exponent : float, optional
        ``M`` of the same spectrum. Must exceed 1.5. Default 2.0.
    spectrum : {'gaussian', 'goff-jordan'}, optional
        ``'goff-jordan'`` adds option ``'g'``. Default ``'gaussian'``.
        This is the ROUGHNESS power spectrum the scattering integral runs
        over. Not to be confused with :class:`~uacpy.models.Scooter`'s
        ``spectrum=``, which is FLP Opt(2) and selects the WAVENUMBER
        spectrum half (``'positive'`` / ``'negative'`` / ``'both'``). The two
        share a name and nothing else.
    rms_roughness : float, optional
        ``|RG|`` (m) at the scattering interface. ``None`` takes the
        environment's own value for that interface.
    interface : int, optional
        Cross-check on the OASES layer index that scatters. OASSP reads it
        from the ``.rhs`` (``unoassp30.f:546-547``); a value that disagrees
        with the file raises rather than attaching the roughness spectrum to
        a layer the binary will not look at.
    realization : int, optional
        Realization index ``k``. The OASES seed is ``-123 - k``
        (``unoassp30.f:170``, ``:535``), so a given ``k`` is reproducible and
        different ``k`` are different draws. Default 0.
    scattered_only : bool, optional
        Option ``'s'`` — zero the source arrays so the ``.trf`` holds the
        scattered field alone (``unoassp30.f:628-635``). Default True.
    multiple_scattering : bool, optional
        **Refused.** OASS supports this; OASSP does not. The vendor disabled
        re-scattering in OASSP's integration path on 980402 — all four
        routines ``unoassp30.f:677-688`` dispatches to
        (``oasvun31.f:76-80``, ``:336-340``, ``:962-966``, ``:1367-1371``)
        have the ``if (.not.rescat)`` guard commented out with the
        ``ROUGH2(II)=0`` zeroing left live, so the option letter is parsed and
        then has no effect. Passing True raises rather than returning a
        single-scatter answer under a multiple-scattering label.
    plane_geometry : bool, optional
        Option ``'P'``. Without it OASSP forces ``CMAX = 1e12`` and a full
        Hankel transform (``unoassp30.f:205-217``). Default False.
    mean_field : OASP, optional
        The producer run. ``None`` builds one from ``n_time_samples`` /
        ``freq_min`` / ``freq_max`` / ``center_frequency``; passing one makes
        those exclusive with it, since the mean field owns the FFT grid.
    n_time_samples, freq_min, freq_max, center_frequency : optional
        Passed to the mean-field :class:`OASP`. OASSP's own Block VIII is then
        read back out of the ``.rhs``, never guessed — see Notes.
    options : str, optional
        Raw OASSP option string, exclusive with the typed flags above.
        ``None`` derives the letters.
    range_start, integration_offset, nw_samples, c_low, c_high : optional
        As for :class:`OASP`. ``c_high`` is inert without ``plane_geometry``.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Supports ``RunMode.BROADBAND`` (complex ``H(f)``, the default) and
    ``RunMode.TIME_SERIES``.

    **Exactly one interface may be rough, and it must be a bottom one.**
    ``oassp2`` reads the mean field's ``.rhs`` back one record per
    wavenumber with no interface filter and takes its scattering interface
    from the first record (``oasvun31.f:66-70``, ``unoassp30.f:549``), so a
    rough sea surface — whose records SCTRHS writes first — or a second
    rough bottom interface hands it interleaved records it cannot use.
    ``run()`` raises up front for either; :class:`OASS`, whose binary
    filters records by the deck interface, handles the multi-interface
    cases.

    **Block VIII is not the user's to set.** OASSP replaces its deck's
    ``NT``/``FR1``/``FR2``/``DT`` with the ``.rhs``'s own values and warns on
    only two of the four (``unoassp30.f:181-188``), so a band that differs
    from the mean field's is substituted with no message at all. This wrapper
    reads the four straight out of the ``.rhs``
    (:func:`~uacpy.io.oases_reader.read_oases_rhs_header`) and writes those,
    which makes the substitution a no-op. Ask for a different band by
    configuring the *mean field*, or by passing ``run(frequencies=…)``, which
    is forwarded to it.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom_range': 'median'``.

    Examples
    --------
    >>> from uacpy.models import OASSP
    >>> oassp = OASSP(correlation_length=5.0, spectral_exponent=2.5,
    ...               n_time_samples=512, freq_min=400, freq_max=600)
    >>> h_scattered = oassp.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASSP:
    # range-independent scattered-field realizations over the same layered
    # stack OASP solves the mean field on — INENVI is shared verbatim between
    # the binaries, so the supported axes are OASP's.
    spec = ModelSpec(
        modes=(RunMode.BROADBAND, RunMode.TIME_SERIES),
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
    )
    source = 'oases'

    #: Constructor knobs that build the mean-field OASP. Named so the
    #: mutual-exclusion message with ``mean_field=`` can list them.
    _MEAN_FIELD_KNOBS = ('n_time_samples', 'freq_min', 'freq_max',
                         'center_frequency')

    def __init__(
        self,
        executable: Optional[Path] = None,
        correlation_length: Optional[float] = None,
        spectral_exponent: float = 2.0,
        spectrum: str = 'gaussian',
        rms_roughness: Optional[float] = None,
        interface: Optional[int] = None,
        realization: int = 0,
        scattered_only: bool = True,
        multiple_scattering: bool = False,
        plane_geometry: bool = False,
        mean_field: Optional['OASP'] = None,
        n_time_samples: Optional[int] = None,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        center_frequency: Optional[float] = None,
        options: Optional[str] = None,
        range_start: Optional[float] = None,
        integration_offset: float = 0.0,
        nw_samples: int = -1,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        self.correlation_length = (
            float(correlation_length) if correlation_length is not None
            else None
        )
        self.spectral_exponent = float(spectral_exponent)
        # Normalised on the way in, so every later comparison — the option
        # letter and the options-exclusivity check below — is exact and
        # spectrum='Gaussian' behaves as the default rather than as a pinned
        # non-default. OASS normalises identically.
        self.spectrum = str(spectrum).lower()
        self.rms_roughness = (
            float(rms_roughness) if rms_roughness is not None else None
        )
        self.interface = int(interface) if interface is not None else None
        self.realization = int(realization)
        self.scattered_only = bool(scattered_only)
        if multiple_scattering:
            raise UnsupportedFeatureError(
                'OASSP',
                "multiple_scattering (option 'p') — the vendor disabled "
                "re-scattering in OASSP's integration path on 980402: every "
                "routine unoassp30.f:677-688 dispatches to "
                "(oasvun31.f:76-80, :336-340, :962-966, :1367-1371) has the "
                "`if (.not.rescat)` guard commented out and the "
                "ROUGH2(II)=0 zeroing left live, so the letter is accepted "
                "and does nothing. OASS still honours it — oassun26.f:491, "
                ":685 and :901 all test .not.rescat there.",
                ['OASS'],
            )
        self.multiple_scattering = False
        self.plane_geometry = bool(plane_geometry)
        self.mean_field = mean_field
        self.n_time_samples = (
            int(n_time_samples) if n_time_samples is not None else None
        )
        self.freq_min = float(freq_min) if freq_min is not None else None
        self.freq_max = float(freq_max) if freq_max is not None else None
        self.center_frequency = (
            float(center_frequency) if center_frequency is not None else None
        )
        self.options = options
        self.range_start = (
            float(range_start) if range_start is not None else None
        )
        self.integration_offset = float(integration_offset)
        self.nw_samples = int(nw_samples)
        self.c_low = float(c_low) if c_low is not None else None
        self.c_high = float(c_high) if c_high is not None else None
        # Same read gate as OASP (unoassp30.f:135-142, OFFDB=OFFDBIN at
        # :375). ``_resolve_options`` always emits 'J' unless a raw string
        # replaces the line.
        _warn_offset_ignored_under_auto_sampling(
            'OASSP', self.integration_offset, self.nw_samples,
            options=self._resolve_options(), offset_letters='Jd')

        if self.correlation_length is None:
            raise ConfigurationError(
                "OASSP requires correlation_length: the roughness power "
                "spectrum is what it integrates, and OASES reads CL from this "
                "deck rather than adopting it from the mean-field run "
                "(oass.tex:182-183).",
                remediation="Pass correlation_length in metres, e.g. "
                            "OASSP(correlation_length=5.0).",
            )
        if self.correlation_length < 0.0:
            raise UnsupportedFeatureError(
                'OASSP',
                "volume scattering (correlation_length < 0 is OASES' switch "
                "to the twelve-token layer record whose extra fields — SKW, "
                "M, RMS, GAM — have no home on any uacpy carrier; "
                "oaseun31.f:76-90)",
                alternatives=["interface roughness scattering "
                              "(correlation_length > 0)"],
                alternatives_label='configurations',
            )
        if self.correlation_length == 0.0:
            raise ConfigurationError(
                "OASSP(correlation_length=0) gives a degenerate roughness "
                "power spectrum: OASSP forms r_l = |CLEN(INTFCE)| and hands "
                "it to PV (unoassp30.f:606-614).",
                remediation="Pass correlation_length > 0 (metres).",
            )
        _reject_unintegrable_spectral_exponent('OASSP', self.spectral_exponent)
        if self.spectrum not in _OASS_SPECTRA:
            raise ConfigurationError(
                f"OASSP(spectrum={spectrum!r}) is not a roughness "
                f"spectrum OASES offers; GETOPT has one letter, 'g' for "
                f"Goff-Jordan (unoassp30.f:1034), and the default is "
                f"Gaussian.",
                remediation=f"Pass one of {sorted(_OASS_SPECTRA)}.",
            )
        if self.mean_field is not None:
            pinned = [n for n in self._MEAN_FIELD_KNOBS
                      if getattr(self, n) is not None]
            if pinned:
                raise ConfigurationError(
                    f"OASSP: mean_field= owns the FFT grid, so "
                    f"{', '.join(pinned)} would be discarded silently — "
                    f"OASSP reads NT/FR1/FR2/DT back out of the .rhs the "
                    f"mean field wrote (unoassp30.f:181-188). Set them on "
                    f"the mean-field model instead.",
                    remediation=(f"OASSP(mean_field=OASP("
                                 f"{'=…, '.join(pinned)}=…))."),
                )
        if options is not None:
            pinned = [n for n, off in (('spectrum', 'gaussian'),
                                       ('scattered_only', True),
                                       ('plane_geometry', False))
                      if getattr(self, n) != off]
            if pinned:
                raise ConfigurationError(
                    f"OASSP: options={options!r} replaces the whole option "
                    f"line, so {', '.join(pinned)} would be discarded "
                    f"silently. Pass either the raw string or the typed "
                    f"flags, not both."
                )

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        # install.sh:1157 copies the raw executables; bin/oases carries
        # symlinks for oasn/oasp/oasr/oast but not for this one, so the
        # name is 'oassp2', not 'oassp'.
        self._exe = self._resolve_executable(
            executable, lambda: _oases_find_executable(self, 'oassp2'),
        )

    def _wavenumber_echo_scale(self) -> int:
        """2 in plane geometry, where the echo under-reports the real grid.

        ``unoassp30.f:327`` prints AUTSAM's count, and the plane-geometry
        branch immediately below (``:337-341``) doubles it — "scattering
        kernels not symmetric", so the integration also runs over negative
        wavenumbers: ``nwvno = 2*nwvno`` with ``ICUT2 = NWVNO``. An echoed
        40000 is 80000 integrated, already past ``NP``, so a guard reading
        the echo verbatim returns silently on exactly the runs it exists to
        catch. Cylindrical geometry keeps ``icut2 = nwvno`` and needs no
        correction.
        """
        return 2 if 'P' in self._resolve_options() else 1

    def _resolve_options(self) -> str:
        """OASSP option letters from the typed flags, or the raw string."""
        if self.options is not None:
            return self.options
        letters = ['N', 'J']
        if self.scattered_only:
            letters.append('s')
        if self.spectrum == 'goff-jordan':
            letters.append('g')
        if self.plane_geometry:
            letters.append('P')
        return ' '.join(letters)

    def _reject_unreadable_options(self) -> None:
        """Reject raw option letters whose ``.trf`` uacpy cannot read back.

        unoassp30.f carries the same complex-frequency-contour trap as OASP:
        ``:285-290`` sets CFRFL under automatic wavenumber sampling unless
        ``'J'`` keeps ICNTIN = 1, ``:983`` sets it for ``'O'``, and
        ``:382-385`` then bakes OMEGIM = -ln(50)·Δf into the spectrum, which
        the time-series synthesis does not undo — the last output sample
        comes back 50× (34 dB) too small. Only a caller-supplied
        ``self.options`` string can reach this: ``_resolve_options``' typed
        path always carries ``'J'`` and never ``'O'``.
        """
        if self.options is None:
            return
        opt_chars = set(str(self.options)) - set(' \t\n')
        if 'O' in opt_chars:
            raise ConfigurationError(
                "OASSP.run: option 'O' (complex frequency integration) bakes "
                "an exp(-ln(50)·Δf·t) contour into the spectrum "
                "(unoassp30.f:983, :382-385) that uacpy's time-series "
                "synthesis does not undo. Drop 'O'."
            )
        if 'J' not in opt_chars and self.nw_samples < 1:
            raise ConfigurationError(
                "OASSP.run: a custom options string without 'J' enables the "
                "complex frequency contour (OMEGIM≠0) under automatic "
                "wavenumber sampling (unoassp30.f:285-290, :382-385), which "
                "uacpy's time-series synthesis cannot undo. Add 'J' or pin "
                "nw_samples >= 1."
            )

    def _build_mean_field(self) -> 'OASP':
        """The OASP producer, with option ``'s'`` guaranteed.

        Without ``'s'`` OASP never calls ``opfilb(45)`` (``unoasp22.f:251-254``)
        and no ``.rhs`` is written at all, so the letter is plumbing rather
        than a physics choice and is added rather than demanded of the caller.
        """
        mean = self.mean_field
        if mean is None:
            kwargs = {k: getattr(self, k) for k in self._MEAN_FIELD_KNOBS
                      if getattr(self, k) is not None}
            mean = OASP(options='N J s', **kwargs)
        if 's' not in set(str(mean.options or 'N J')) - set(' \t\n'):
            mean = mean.copy(options=f"{mean.options or 'N J'} s")
            self._log("Added option 's' to the mean-field model: without it "
                      "OASP writes no .rhs (unoasp22.f:251-254)")
        return mean

    def _run_mean_field(self, env, source, receiver, fm, frequencies):
        """Run the mean field into ``fm.work_dir``; return its Result and stem.

        The ``.rhs`` filename reaching the binary is relative —
        ``_oases_subprocess_env`` builds ``f'{base_name}.045'`` and
        ``_run_and_attach_prt`` runs with ``cwd=work_dir`` — so the producer's
        output must sit in the same directory as the OASSP deck. ``copy()``
        keeps a user-supplied ``mean_field``'s own configuration and redirects
        only the plumbing; ``use_tmpfs=False`` avoids the "ignored for the
        pinned work_dir" warning ``_setup_file_manager`` would otherwise emit,
        the outer ``fm`` having honoured it already.
        """
        mean = self._build_mean_field().copy(
            work_dir=fm.work_dir, cleanup=False, use_tmpfs=False,
            verbose=self.verbose, timeout=self.timeout,
        )
        stem = 'oasp_run'
        self._log(f"Running {mean.model_name} for the mean field (option 's')")
        mean_result = mean.run(env, source, receiver, RunMode.BROADBAND,
                               frequencies=frequencies)
        self._require_output(
            [fm.get_path(f'{stem}.045')],
            what="a mean-field .rhs (option 's')",
            hint=("OPFILB opens unit 45 with STATUS='UNKNOWN' "
                  "(oashun21.f:634), so an empty file here means the producer "
                  "wrote no scattering right-hand sides."),
        )
        self._require_output(
            [fm.get_path(f'{stem}.046')],
            what="a mean-field .vol (FOR046)",
            hint=("OASSP opens unit 46 unconditionally (unoassp30.f:128) and "
                  "its IOER test only covers the last OPFILb (:130)."),
        )
        return mean_result, stem

    def _require_single_rough_interface(self, env) -> None:
        """Refuse environments whose ``.rhs`` OASSP cannot consume, before a
        mean-field run is spent producing it.

        ``SCTRHS`` writes one ``.rhs`` record per rough interface per
        wavenumber, the sea surface's first (``oaseun31.f:2306-2310``,
        ``:2395``), and ``oassp2`` reads back exactly one record per
        wavenumber with **no interface filter**, taking its scattering
        interface from the first record (``unoassp30.f:549``,
        ``oasvun31.f:66-70``) — unlike OASS, which filters records by its
        deck's ``INTFC`` (``oassun26.f:360-362``). Measured on a rough
        surface + rough bottom: the binary announces ``Roughness scattering
        Layer: 2`` (the sea surface, whose water record carries no roughness
        spectrum), derives a zero wavenumber step from the interleaved
        records (``dkr_i= (0,0)``) and stops on ``>>> ERROR: Frequency
        mismatch in rhs file <<<``, leaving a truncated ``.trf``; two rough
        *bottom* interfaces fail identically. A mean field with no rough
        interface writes a ``.rhs`` with no SCTRHS records at all (skipped
        below ``ROUGH2 < 1e-10``, ``oaseun31.f:2310``), equally unusable.
        So exactly one interface may be rough, and it must be a bottom one.
        """
        surface_rg = abs(float(
            getattr(env.surface, 'roughness', 0.0) or 0.0))
        rough_bottoms = [rg for rg in bottom_interface_roughness(env)
                         if abs(rg) > _OASS_MIN_MEAN_FIELD_ROUGHNESS]
        if surface_rg > _OASS_MIN_MEAN_FIELD_ROUGHNESS:
            raise UnsupportedFeatureError(
                'OASSP',
                f"a rough sea surface (env.surface.roughness = "
                f"{surface_rg:g} m) — OASSP scatters from the seabed: the "
                f"mean field writes the surface's boundary operators into "
                f"the .rhs first, and oassp2 takes its scattering interface "
                f"from that first record and reads one record per wavenumber "
                f"with no interface filter, so the run scatters from the "
                f"water record's empty roughness spectrum and dies on "
                f"interleaved records",
                ['OASS (its binary filters .rhs records by the deck '
                 'interface, so it handles a rough surface alongside a '
                 'rough bottom)',
                 'a smooth surface (env.surface.roughness <= 1e-5 m) with '
                 'the roughness on the seabed'],
                alternatives_label='configurations',
            )
        if not rough_bottoms:
            raise ConfigurationError(
                "OASSP: every bottom interface is smooth in the environment, "
                "so the mean-field run would write a .rhs with no boundary-"
                "operator records — SCTRHS skips any interface with "
                "ROUGH2 < 1e-10 (oaseun31.f:2310, :379) — and OASSP would "
                "have nothing to scatter from.",
                remediation=("Set the roughness on the environment, e.g. "
                             "BoundaryProperties(..., roughness=0.5). "
                             "OASSP(rms_roughness=…) overrides only the "
                             "OASSP deck, not the mean field's."),
            )
        if len(rough_bottoms) > 1:
            raise UnsupportedFeatureError(
                'OASSP',
                f"{len(rough_bottoms)} rough bottom interfaces — SCTRHS "
                f"writes one .rhs record per rough interface per wavenumber "
                f"and oassp2 reads back exactly one per wavenumber, so the "
                f"interleaved records give it a zero wavenumber step and "
                f"the run dies with a truncated .trf",
                ['OASS (its binary filters .rhs records by the deck '
                 'interface)',
                 'an environment with exactly one rough bottom interface'],
                alternatives_label='configurations',
            )

    def _resolve_interface(self, env, rhs_header: dict) -> int:
        """The OASES layer index whose roughness OASSP will scatter from.

        ``SCTRHS`` loops over interfaces innermost (``oaseun31.f:2306-2310``)
        and OASSP reads the *first* record's index (``unoassp30.f:546-547``);
        ``_require_single_rough_interface`` has already guaranteed the mean
        field wrote exactly one rough interface, a bottom one, so that first
        record names the seabed being scattered from.
        """
        from_file = int(rhs_header['interface'])
        if self.interface is not None and self.interface != from_file:
            raise ConfigurationError(
                f"OASSP(interface={self.interface}) disagrees with the "
                f"mean-field .rhs, which names interface {from_file} "
                f"(unoassp30.f:546-547 reads it from the file, not the deck). "
                f"Writing the roughness spectrum onto layer "
                f"{self.interface} would leave layer {from_file} with "
                f"CLEN = 0 (oaseun31.f:102) and the run would exit 0 with no "
                f"back-scatter.",
                remediation=(f"Drop interface=, or set it to {from_file}."),
            )
        return from_file

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """
        Run the OASP → OASSP chain and return the scattered field.

        Parameters
        ----------
        env : Environment
            Ocean environment (range-independent; range-dependent features
            are collapsed with a warning). The scattering interface must
            carry a non-zero roughness.
        source : Source
            Acoustic source. Shared by both runs, as OASSP's geometry must
            match the mean field's.
        receiver : Receiver
            Receiver array; ``ranges`` must be uniformly spaced.
        run_mode : RunMode, optional
            ``BROADBAND`` (default) — complex scattered ``H(f)``.
            ``TIME_SERIES`` — real scattered pressure p(t); requires
            ``source_waveform`` + ``sample_rate``.
        frequencies : ndarray, optional
            Forwarded to the **mean-field** model, which owns the FFT grid.
        source_waveform, sample_rate, output_duration : optional
            As for :meth:`OASP.run`.

        Returns
        -------
        result : Field
            Complex ``H(f)`` on ``(depth, range, frequency)`` for
            ``BROADBAND``; real p(t) on ``(depth, range, time)`` for
            ``TIME_SERIES``.
        """
        self._require_run_triple(env, source, receiver)
        run_mode = self._resolve_run_mode(run_mode)
        source_waveform, frequencies = self._prepare_timeseries(
            run_mode, source, frequencies, source_waveform, sample_rate,
            output_duration,
        )

        options = self._resolve_options()
        self._reject_unreadable_options()
        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        self._require_single_rough_interface(env)
        fm = self._setup_file_manager()

        try:
            mean_result, mean_stem = self._run_mean_field(
                env, source, receiver, fm, frequencies)
            rhs_path = fm.get_path(f'{mean_stem}.045')
            vol_path = fm.get_path(f'{mean_stem}.046')
            rhs = read_oases_rhs_header(rhs_path)
            interface = self._resolve_interface(env, rhs)

            base_name = 'oassp_run'
            input_file = fm.get_path(f'{base_name}.dat')
            self._log(f"Writing OASSP input file: {input_file} "
                      f"(options={options}, realization={self.realization}, "
                      f"interface={interface})")
            write_oassp_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                options=options,
                interface=interface,
                correlation_length=self.correlation_length,
                spectral_exponent=self.spectral_exponent,
                rms_roughness=self.rms_roughness,
                realization=self.realization,
                # Block VIII, taken from the file the binary itself will read
                # rather than re-derived, so unoassp30.f:181-188 substitutes
                # the values that are already there.
                n_time_samples=rhs['n_time_samples'],
                freq_min=rhs['freq_min'],
                freq_max=rhs['freq_max'],
                time_step=rhs['time_step'],
                # Block III's FRC rides into the .trf header (TRFHEAD's FREQS
                # argument, oasiun23.f:815, written to the header by
                # oasiun23.f:864-865,919), so take the carrier the mean field
                # actually used rather than a second resolution of it.
                center_frequency=mean_result.metadata.get(
                    'center_frequency', self.center_frequency),
                integration_offset=self.integration_offset,
                nw_samples=self.nw_samples,
                c_low=self.c_low,
                c_high=self.c_high,
                range_start=self.range_start,
            )

            # FOR045/FOR046 point oassp2 at the mean-field run's kernel
            # files, named after that run's stem (see OASES._execute).
            proc = self._execute(base_name, fm.work_dir, extra_env={
                'FOR045': f'{mean_stem}.045', 'FOR046': f'{mean_stem}.046',
            })
            # unoassp30.f has no NWVNO.GT.NP test either (see the base
            # method), and its reference range min(cref·NX·DT + RM, 6·RM)
            # (:295-306) can push NW ~2× past the mean-field OASP's, so the
            # producer passing is no proxy for this run being in bounds.
            self._reject_wavenumber_overrun(proc)
            output_file = self._require_output(
                [fm.get_path(f'{base_name}.trf')], what='a transfer-function file',
                process=proc,
            )
            self._log(f"Reading OASSP output: {output_file}")
            trf_data = read_oasp_trf(output_file,
                                     receiver_depths=receiver.depths)

            # Convention: (n_depth, n_range, n_frequencies) — trailing axis is
            # the variable dim. Source axes: (freq, range, depth). Negated
            # like OASP's: the raw payload comes from the same TRFHEAD
            # writer with the same sign (measured with scattered_only=False
            # on a near-smooth seabed, the raw total field over OASP's raw
            # mean field is +1: |ratio| 0.995, arg 0.35 deg), so the same
            # negation is what makes 'travelling_wave' mean the family's
            # convention here too.
            tf_reordered = -np.transpose(
                trf_data['transfer_function'], (2, 1, 0))

            result = Field(
                data=tf_reordered,
                coords={
                    'depth': trf_data['depths'],
                    'range': trf_data['ranges'],
                    'frequency': trf_data['freq'],
                },
                # The same complex H(f) OASP returns, from the same TRFHEAD
                # writer (oasiun23.f:842-845).
                **self._result_kwargs(
                    source,
                    phase_reference='travelling_wave',
                    backend='oassp',
                    frequencies=trf_data['freq'],
                    source_depth=trf_data['source_depth'],
                    center_frequency=trf_data['center_frequency'],
                    realization=self.realization,
                    interface=interface,
                    n_time_samples=rhs['n_time_samples'],
                    freq_max=rhs['freq_max'],
                    mean_field_result=mean_result,
                ),
            )
            result = _mask_zero_range(result, source, 'OASSP')
            if run_mode == RunMode.TIME_SERIES:
                result = result.synthesize_time_series(
                    source_waveform=source_waveform,
                    sample_rate=sample_rate,
                )

            self._attach_output_paths(
                result, fm.work_dir, base_name,
                primary_files=(('trf_file', '.trf'),),
            )
            if not self.cleanup:
                result.metadata['rhs_file'] = str(rhs_path)
                result.metadata['vol_file'] = str(vol_path)

            self._log("OASSP simulation complete")
            return result

        finally:
            fm.finish()

    # Mirrors third_party/oases/bin/oassp (oassp.tex:524-536). The .trf is
    # named from the deck stem by TRFHEAD (oasiun23.f:842-845), not by an
    # environment variable, so it needs no unit here; FOR045/FOR046 are set
    # per-run in _execute because they name the mean-field deck's stem.
    # Units 68-71 and 85 are unconditional ASCII dumps of the realized
    # perturbation field (oasvun31.f:421-448, :724), and dum.dum is the
    # scratch file itran() shells out for under option 'r' (:1190-1210) —
    # all listed so a pinned work_dir starts each run clean. The bare
    # fort.46 is absent because oassp2 neither writes nor reads it: units
    # 45/46 are assigned to the mean-field stem's .045/.046 above, which are
    # different files, and nothing here opens the default name.
    _FOR_FILES: dict = {}
    _OUTPUT_SUFFIXES = ('.trf', '.plt', '.plp')
    _OUTPUT_FORT_FILES = ('fort.68', 'fort.69', 'fort.70', 'fort.71',
                          'fort.85', 'dum.dum')


class OASS(OASES):
    """
    OASS — OASES Scattering and Reverberation Module.

    Computes the spatial statistics of the field a rough interface scatters:
    the reverberation level against range and receiver depth
    (``RunMode.REVERBERATION``, the default) or the reverberation covariance
    across the receiver array (``RunMode.COVARIANCE``).

    OASS is a **post-processor**. It consumes the mean-field boundary
    operators an OAST run writes with option ``'s'`` into a ``.rhs``
    file, and under the reverberation options it re-derives its whole
    wavenumber axis from them (``unoass21.f:197-216``). ``run()`` therefore
    drives two binaries in one work dir: the mean-field model first, then
    ``oass2``. The mean-field ``Result`` is kept on
    ``metadata['mean_field_result']`` so the coherent field is not thrown
    away.

    Parameters
    ----------
    executable : Path, optional
        Path to the ``oass2`` binary. Auto-detected if ``None``.
    correlation_length : float
        ``CL`` (m), the roughness correlation length of the scattering
        interface. **Required**: ``oass.tex:182-183`` — the rms roughness,
        correlation length and spectral exponent "must be specified. These
        are not adopted from OASR or OAST."
    spectral_exponent : float
        ``M``, the roughness power-spectrum exponent. Must exceed 1.5 or the
        spectrum is not integrable. Default 2.0.
    spectrum : str
        ``'gaussian'`` (default) or ``'goff-jordan'`` (option ``'g'``,
        ``unoass21.f:674``).
    rms_roughness : float, optional
        ``|RG|`` (m) at the scattering interface in the **OASS** deck.
        ``None`` takes the environment's own roughness there. The mean-field
        deck always uses the environment's value — ``oass.tex:184-186``
        allows the two to differ.
    interface : int, optional
        ``INTFC``, the OASES deck-layer index of the scattering interface
        (``unoass21.f:151``). ``None`` selects the seafloor, i.e. the first
        bottom record of the deck.
    multiple_scattering : bool
        Option ``'p'`` (``rescat``): perturbed boundary operator. Yields a
        lower bound on the reverberation level where the default single-
        scattering kernel yields an upper bound (``oass.tex:163-166``).
    plane_geometry : bool
        Option ``'P'`` (``ICDR=1``): plane rather than cylindrical geometry.
    mean_field : OAST, optional
        The producer run. ``None`` builds ``OAST(options='N J T s')``. A
        supplied model keeps its own configuration; only ``work_dir``,
        ``cleanup``, ``use_tmpfs``, ``verbose`` and ``timeout`` are
        redirected onto this run. An ``OASR`` is refused: its ``.rhs``
        stamps every record with interface index 2 and samples wavenumber
        as ``cos(angle)``, neither of which the ``INTFC``-match /
        uniform-``DLWVNO`` inference at ``unoass21.f:200-211`` can consume.
    c_low, c_high : float, optional
        ``CMIN``/``CMAX`` (m/s). ``c_low`` is **physically significant, not a
        tuning knob**: it truncates the scattering integral. Measured on one
        ``.rhs``, raising it from 1350 to 2000 m/s moved the answer by
        30.15 dB while lowering it to 900 m/s changed nothing (1e-4 dB),
        because ``REVINT``/``REVCOV`` bound their own reads of the mean
        field's wavenumber buffer (``oassun26.f:744-748``, ``:958-962``).
        ``None`` takes the mean field's own bound, and a value that differs
        from it warns.
    receiver_gains : ndarray, optional
        Per-element gain (dB), Block VI column 5. Scalar or one per element.
    options : str, optional
        Raw OASS option letters, written verbatim; ``None`` derives them from
        ``run_mode`` plus ``spectrum`` / ``multiple_scattering`` /
        ``plane_geometry``. Combining a raw string with any of those three
        raises ``ConfigurationError`` — the string replaces the whole option
        line, so a flag passed alongside it would be discarded.
    integration_offset, nw_samples
        Accepted only to refuse them: neither reaches this binary. See
        ``__init__``.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    **There is no field-parameter option letter in OASS** and ``'R'`` means
    *reverberant field*, not radial stress — see ``write_oass_input``'s
    docstring for the manual/source disagreements the deck rides on.

    One product per run: ``'a'`` on the same option line as ``'r'`` returns a
    **silently zero** covariance, because ``REVINT`` zeroes ``ROUGH2`` for
    every layer before ``REVCOV`` reads it (``oassun26.f:683-688`` vs
    ``:897-902``). The run mode picks exactly one letter, and the writer
    refuses any deck that asks for two.

    ``RunMode.REVERBERATION`` returns a real dB :class:`Field` tagged
    ``kind='reverberation'`` — ``-10·log10 E[|p_scat|²]``, **not**
    transmission loss, so it does not compare against a TL field.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).** Per-model:
    ``'ssp': 'mean'``, ``'bottom_range': 'median'`` (the layer stack is
    kept).

    Examples
    --------
    >>> from uacpy.models import OASS
    >>> oass = OASS(correlation_length=10.0, rms_roughness=0.5)
    >>> reverb = oass.run(env, source, receiver)
    >>> cov = oass.compute_covariance(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASS:
    # range-independent reverberation from one rough interface; multi-layer
    # fluid + elastic bottom honoured. Single spectral solve → mean SSP /
    # median bottom column.
    spec = ModelSpec(
        modes=(RunMode.REVERBERATION, RunMode.COVARIANCE),
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
    )
    source = 'oases'

    #: The one product letter each run mode emits (see the Notes above).
    _PRODUCT_LETTER = {
        RunMode.REVERBERATION: 'r',
        RunMode.COVARIANCE: 'a',
    }

    def __init__(
        self,
        executable: Optional[Path] = None,
        # Roughness statistics of the scattering interface (Block IV).
        correlation_length: Optional[float] = None,
        spectral_exponent: float = 2.0,
        spectrum: str = 'gaussian',
        rms_roughness: Optional[float] = None,
        interface: Optional[int] = None,
        multiple_scattering: bool = False,
        plane_geometry: bool = False,
        # Mean-field producer. ``None`` builds OAST(options='N J T s').
        mean_field: Optional['OASES'] = None,
        # Wavenumber bounds (Block VII line 1).
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        # Array element gains (Block VI column 5), dB.
        receiver_gains: Optional[np.ndarray] = None,
        options: Optional[str] = None,
        integration_offset: float = 0.0,
        nw_samples: Optional[int] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        if correlation_length is None:
            raise ConfigurationError(
                "OASS(correlation_length=…) is required: the roughness power "
                "spectrum P(k) is what the reverberation integral integrates "
                "over, and oass.tex:182-183 states the rms roughness, "
                "correlation length and spectral exponent are not adopted "
                "from the OAST/OASR mean-field run.",
                remediation=("Pass the interface correlation length in "
                             "metres, e.g. OASS(correlation_length=10.0)."),
            )
        self.correlation_length = float(correlation_length)
        self.spectral_exponent = float(spectral_exponent)
        _reject_unintegrable_spectral_exponent('OASS', self.spectral_exponent)
        if str(spectrum).lower() not in _OASS_SPECTRA:
            raise ConfigurationError(
                f"OASS(spectrum={spectrum!r}) is not a roughness spectrum "
                f"OASS implements; valid are "
                f"{sorted(_OASS_SPECTRA)} — GETOPT's only spectrum letter is "
                f"'g'/'G' (goff, unoass21.f:674) and its absence means "
                f"Gaussian."
            )
        # Normalised so the option letter and the options-exclusivity check
        # compare exactly, and so OASS and OASSP accept the same spellings.
        self.spectrum = str(spectrum).lower()
        self.rms_roughness = (
            float(rms_roughness) if rms_roughness is not None else None
        )
        self.interface = int(interface) if interface is not None else None
        self.multiple_scattering = bool(multiple_scattering)
        self.plane_geometry = bool(plane_geometry)
        if isinstance(mean_field, OASR):
            raise ConfigurationError(
                "OASS(mean_field=OASR(...)) divides by zero inside oass2. "
                "OASS keeps only the .rhs records whose interface index "
                "matches INTFC (unoass21.f:200), and takes DLWVNO from the "
                "gap between the first two it keeps (:207) before "
                "NWVNO=(WKMAX-WK0)/DLWVNO+1 at :211. Two things stop that "
                "from working on an OASR .rhs. Its layer 1 is the water "
                "half-space, so SCTRHS stamps IN1=2 (oaseun31.f:2308-2309, "
                "IN1=IN+1) on every record, while INTFC here is a bottom "
                "deck layer, 3 or deeper — no record matches, KCNT stays 0 "
                "and DLWVNO is never assigned. And OASR calls SCTRHS from "
                "inside its angle loop with WVNO = Re(AK(1,1))*cos(ANG) "
                "(oasjun21.f:70,108), so even a matching record set is "
                "cosine-spaced in wavenumber, not the uniform grid the "
                "DLWVNO-from-two-records inference assumes. oass.tex:18-23 "
                "names OASR as a producer, but only for a deck whose "
                "scattering interface is its layer 2.",
                remediation="Pass an OAST whose option line has 's' (the "
                            "default mean field), or None.",
            )
        if mean_field is not None and not isinstance(mean_field, OAST):
            raise ConfigurationError(
                f"OASS(mean_field={type(mean_field).__name__}) cannot produce "
                f"the boundary operators OASS consumes: oass.tex:18-23 names "
                f"OAST and OASR as the producers, and OASS reads exactly one "
                f"per-frequency header record from the .rhs "
                f"(unoass21.f:123).",
                remediation="Pass an OAST whose option line has 's'.",
            )
        self.mean_field = mean_field
        self.c_low = float(c_low) if c_low is not None else None
        self.c_high = float(c_high) if c_high is not None else None
        self.receiver_gains = (
            np.asarray(receiver_gains, dtype=float)
            if receiver_gains is not None else None
        )
        # Raw OASS option string. ``None`` lets the wrapper derive it from the
        # run mode and the typed flags; a raw string replaces the deck's
        # option line outright, so the two ways of specifying it are
        # exclusive.
        self.options = options
        if options is not None:
            pinned = [n for n, unset in (('spectrum', 'gaussian'),
                                         ('multiple_scattering', False),
                                         ('plane_geometry', False))
                      if getattr(self, n) != unset]
            if pinned:
                raise ConfigurationError(
                    f"OASS: options={options!r} replaces the whole option "
                    f"line, so {', '.join(pinned)} would be discarded "
                    f"silently. Pass either the raw string or the typed "
                    f"flags, not both — note the derived string carries the "
                    f"product letter for the run mode ('r' or 'a'), which a "
                    f"raw string must repeat."
                )

        # Both knobs exist on the OAST/OASN/OASP constructors and neither can
        # reach this binary, so they are refused rather than recorded: OASN's
        # precedent for a knob that cannot reach its deck (plot_rmin, vrec).
        if integration_offset:
            raise ConfigurationError(
                "OASS(integration_offset=…) has no effect: Block III's COFF "
                "is read only when ICNTIN > 0, and OASS's GETOPT never sets "
                "it (unoass21.f:596 initialises it to 0 and :557-698 never "
                "assigns). Under the reverberation options the contour offset "
                "is re-derived from the .rhs (unoass21.f:213).",
                remediation=("Set the offset on the mean-field model instead: "
                             "OASS(mean_field=OAST(integration_offset=…))."),
            )
        if nw_samples is not None:
            raise ConfigurationError(
                "OASS(nw_samples=…) has no effect: under every option OASS "
                "supports the binary recomputes NWVNO from the .rhs file's "
                "own wavenumber step and forces ICUT1=1, ICUT2=NWVNO "
                "(unoass21.f:209-215). Measured: a deck asking for 2048 ran "
                "2778.",
                remediation=("Set the sampling on the mean-field model "
                             "instead: OASS(mean_field=OAST(nw_samples=…))."),
            )
        self.integration_offset = 0.0
        self.nw_samples = None

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        # install.sh:1157 copies the raw executable; bin/oases carries
        # symlinks for oasn/oasp/oasr/oast but none for oass.
        self._exe = self._resolve_executable(
            executable, lambda: _oases_find_executable(self, 'oass2'),
        )

    def _resolve_options(self, run_mode: RunMode) -> str:
        """The OASS option line for this run.

        A raw ``options`` string is written verbatim; otherwise the letters
        are the run mode's product letter plus the typed flags. The
        constructor rejects the two being combined.
        """
        if self.options is not None:
            return self.options
        opt = [self._PRODUCT_LETTER[run_mode]]
        if self.spectrum == 'goff-jordan':
            opt.append('g')
        if self.multiple_scattering:
            opt.append('p')
        if self.plane_geometry:
            opt.append('P')
        return ' '.join(opt)

    def _reject_unreadable_options(self, options: str,
                                   run_mode: RunMode) -> None:
        """Refuse raw option letters whose output uacpy cannot read back.

        OASES scans the option line character by character
        (``unoass21.f:599`` ``READ(1,200) OPT`` / ``FORMAT(40A1)``), so the
        tests are on the character set rather than on whitespace-split
        tokens. The deck-level letter validation (unknown letters, two
        products in one deck) belongs to ``write_oass_input`` and is not
        repeated here.
        """
        chars = set(str(options)) - set(' \t\n')

        contours = sorted({'C', 'D'} & chars)
        if contours:
            raise UnsupportedFeatureError(
                'OASS',
                f"option(s) {contours} (horizontal-correlation / "
                f"reverberation-intensity contours) — the output goes to the "
                f"CONDRW/CONDRB pair on units 28/29 (unoass21.f:277-278) and "
                f"uacpy has no reader for that format. 'D' computes the same "
                f"REVINT quantity RunMode.REVERBERATION already returns",
            )
        kernels = sorted({'I', 'S', 'c'} & chars)
        if kernels:
            raise UnsupportedFeatureError(
                'OASS',
                f"option(s) {kernels} (scattering kernels for one incident "
                f"plane wave) — their curves carry the INTGR / SPECT plot "
                f"tags, and read_oast_tl selects on the TLRAN tag "
                f"(io/oases_reader.py, _OAST_TL_RANGE_TAG), so nothing would "
                f"be found in the .plt",
            )
        if 'Z' in chars:
            raise UnsupportedFeatureError(
                'OASS',
                "option 'Z' (SVP plot) — it makes the binary read two further "
                "records after Block IX (unoass21.f:319-320) that "
                "write_oass_input does not emit, so the deck ends early and "
                "OASS reads past it",
            )
        letter = self._PRODUCT_LETTER[run_mode]
        if letter not in chars:
            raise ConfigurationError(
                f"OASS(options={options!r}) does not carry {letter!r}, the "
                f"product letter for run_mode={run_mode.name}: the deck would "
                f"compute something other than what run() then tries to read.",
                remediation=(f"Add {letter!r} to the option string, or run the "
                             f"mode whose product it does carry."),
            )

    def _require_single_frequency(self, source: Source) -> float:
        """The one frequency this chain runs at (G1).

        ``unoass21.f:123`` pins ``nfreq = 1`` before reading the ``.rhs``, so
        OASS reads exactly one per-frequency header record and treats every
        record after it as a 19-word ``SCTRHS`` record. A multi-frequency
        mean-field run therefore ends in a Fortran I/O error rather than a
        clean stop.
        """
        freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
        if freqs.size > 1:
            raise ConfigurationError(
                f"OASS: the mean-field run must be single-frequency, but "
                f"source.frequencies has {freqs.size} entries "
                f"({freqs.min():g}-{freqs.max():g} Hz). OASS pins nfreq=1 "
                f"before reading the .rhs (unoass21.f:123) and reads the "
                f"second frequency's 5-word header as a 19-word boundary-"
                f"operator record.",
                remediation=("Run one frequency at a time: "
                             "Source(depths=…, frequencies=f)."),
            )
        return float(freqs[0])

    def _resolve_interface(self, env: Environment) -> int:
        """``INTFC`` for this environment, defaulting to the seafloor (G3).

        The sea surface, deck layer 2, is a legal target as well as the bottom
        records: ``ROUGH(M)`` is the roughness of the interface at the top of
        layer M (``oaseun31.f:377-383``), ``SCTRHS`` writes its operators under
        ``IN1 = 2`` like any other, and ``write_oass_input`` routes the
        nine-token roughness tail to the water record for it. An ice keel or a
        wind-roughened surface is the usual case, and ``spec.supports``
        advertises ``'rough_surface'``.

        Also checks that the interface is rough in the **mean-field** deck:
        ``SCTRHS`` skips any interface with ``ROUGH2 < 1e-10``
        (``oaseun31.f:2310``, ``ROUGH2 = ROUGH**2`` at ``:379``), so a smooth
        environment produces a ``.rhs`` with no boundary-operator records at
        all — ``KCNT`` stays 0, ``DLWVNO`` is never set and OASS divides by
        zero. The environment's roughness reaches the mean-field deck
        whatever ``rms_roughness`` overrides in the OASS deck.
        """
        first_bottom, roughness = oass_bottom_interfaces(env)
        interface = (first_bottom if self.interface is None
                     else int(self.interface))
        if interface == _OASS_SURFACE_INTERFACE:
            rg = abs(float(getattr(env.surface, 'roughness', 0.0) or 0.0))
            named = "the sea surface (deck layer 2)"
            fix = ("Set the roughness on the surface, e.g. "
                   "BoundaryProperties(..., roughness=0.5) on env.surface. ")
        else:
            index = interface - first_bottom
            if not 0 <= index < len(roughness):
                raise ConfigurationError(
                    f"OASS(interface={interface}) is not a scattering "
                    f"interface of this deck: it carries the sea surface at "
                    f"deck layer {_OASS_SURFACE_INTERFACE} and the seabed "
                    f"stack at layers {first_bottom}.."
                    f"{first_bottom + len(roughness) - 1}.",
                    remediation=("The roughness spectrum can only be attached "
                                 "to the first water record or a bottom layer "
                                 "record; an interior water interface has no "
                                 "RG column INENVI would re-read as CL and M."),
                )
            rg = abs(float(roughness[index]))
            named = f"interface {interface}"
            fix = ("Set the roughness on the environment, e.g. "
                   "BoundaryProperties(..., roughness=0.5). ")
        if rg <= _OASS_MIN_MEAN_FIELD_ROUGHNESS:
            raise ConfigurationError(
                f"OASS: {named} is smooth in the environment "
                f"(roughness = {rg:g} m), so the mean-field run "
                f"writes an empty .rhs — SCTRHS skips any interface with "
                f"ROUGH2 < 1e-10 (oaseun31.f:2310, :379) and OASS then has no "
                f"wavenumber step to integrate on.",
                remediation=(fix + "OASS(rms_roughness=…) overrides only the "
                             "OASS deck, not the mean field's."),
            )
        return interface

    def _mean_field_c_low(self, env: Environment) -> float:
        """``CMIN`` the mean-field deck integrates on.

        OAST and OASR have no ``c_low`` knob — their writers derive it from
        the profile — so this reproduces that derivation, and reads the knob
        off any producer that does carry one.
        """
        pinned = getattr(self.mean_field, 'c_low', None)
        if pinned is not None:
            return float(pinned)
        ssp_data = env.ssp.extend_to(float(env.depth)).to_pairs()
        return oases_wavenumber_bounds(ssp_data)[0]

    def _warn_on_c_low_mismatch(self, env: Environment) -> None:
        """Warn when this deck's ``CMIN`` is not the mean field's (G4).

        Measured on one ``.rhs``, option ``'r'``, only ``CMIN`` changed:
        1350 → 2000 m/s moved the field by 30.15 dB; 1350 → 900 m/s moved it
        by 1e-4 dB. Raising ``c_low`` truncates the scattering integral,
        because ``REVINT``/``REVCOV`` bound their reads of the mean field's
        buffer themselves (``oassun26.f:744-748``, ``:958-962``) — wavenumbers
        past the mean field's grid simply contribute nothing.
        """
        if self.c_low is None:
            return
        mean_c_low = self._mean_field_c_low(env)
        if np.isclose(self.c_low, mean_c_low, rtol=1e-6):
            return
        direction = ('truncates' if self.c_low > mean_c_low
                     else 'does not extend')
        warnings.warn(
            f"OASS(c_low={self.c_low:g}) differs from the mean field's "
            f"{mean_c_low:g} m/s. c_low is physically significant here, not a "
            f"tuning knob: it {direction} the scattering integral over the "
            f".rhs wavenumber grid (measured 30.15 dB for 1350 → 2000 m/s, "
            f"1e-4 dB for 1350 → 900). Leave c_low unset to inherit the mean "
            f"field's bound.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    def _run_mean_field(self, env: Environment, source: Source,
                        receiver: Receiver, fm, frequency: float):
        """Run the producer into ``fm.work_dir``; return its Result and stem.

        The ``.rhs`` it leaves behind is this run's FOR045. The filename
        reaching the binary is relative (``_oases_subprocess_env`` builds
        ``f'{base_name}.045'`` and the child runs with ``cwd=work_dir``), so
        both decks must sit in the same directory — hence one shared file
        manager rather than a scratch dir per binary.
        """
        mean = self.mean_field if self.mean_field is not None else OAST(
            options='N J T s')
        # The option line the producer will write: OAST derives its line
        # from typed flags via _resolve_options; OASR has no resolver — its
        # writer takes ``options`` verbatim (defaulting when None).
        resolve = getattr(mean, '_resolve_options', None)
        mean_options = resolve() if callable(resolve) else (mean.options or '')
        chars = set(mean_options) - set(' \t\n')
        if 's' not in chars:
            raise ConfigurationError(
                f"OASS(mean_field={type(mean).__name__}(options=…)): the "
                f"producer's option line must contain 's' or it writes no "
                f".rhs — SCTOUT gates the whole boundary-operator dump "
                f"(oaseun31.f:1899-1900).",
                remediation="e.g. OASS(mean_field=OAST(options='N J T s')).",
            )
        # The producer runs through its own wrapper, which requires its own
        # primary table: OAST's .plt (FOR020) and OASR's .rco/.trc are both
        # written under 'T'. Without it the producer leaves the .rhs OASS
        # actually needs and then fails on a table nothing here reads, so name
        # the real requirement instead.
        if 'T' not in chars:
            raise ConfigurationError(
                f"OASS(mean_field={type(mean).__name__}(options="
                f"{mean_options!r})): the producer's option line must also "
                f"contain 'T'. OASS consumes only the .rhs, but the mean field "
                f"is run through its own wrapper, which requires the table 'T' "
                f"writes (OAST's FOR020 .plt, OASR's .rco/.trc) and fails "
                f"without it — after the binary has already run.",
                remediation="e.g. OASS(mean_field=OAST(options='N J T s')).",
            )
        mean = mean.copy(work_dir=fm.work_dir, cleanup=False, use_tmpfs=False,
                         verbose=self.verbose, timeout=self.timeout)
        stem = f'{mean.model_name.lower()}_run'
        self._log(f"Running {mean.model_name} for the mean field (option 's')")
        mean_result = mean.run(env, source, receiver)
        rhs = self._require_output(
            [fm.get_path(f'{stem}.045')],
            what="a mean-field .rhs (option 's')",
            hint=("The producer ran but left no boundary operators; check "
                  "that the scattering interface is rough."),
        )

        # G2 — read the contract off the bytes the consumer will see, not off
        # what the producer was asked for. A mismatch is not an abort:
        # unoass21.f:128-135 prints '>>> WARNING: Frequency sampling mismatch'
        # to stdout and then executes ``freq=fr1_in_file``, so the run
        # silently moves to the .rhs's frequency while uacpy would label the
        # result with the deck's. (The '>>> FREQUENCY MISMATCH. ABORTING <<<'
        # test at :143 compares the already-replaced freq against record 2's,
        # which the same producer wrote, so it never fires.)
        #
        # The first record is ``NX, FR1, FR2, DT`` for every producer, but a
        # single-frequency OAST/OASR writes ``nfreq`` into the NX slot and
        # ``1/(nfreq*dlfreq)`` into DT (unoast31.f:164-165, unoasr21.f:222-223)
        # — the same four fields OASS reads back as nx_in_file/fr1_in_file
        # (unoass21.f:127).
        header = read_oases_rhs_header(rhs)
        nfreq = header['n_time_samples']
        rhs_frequency = header['freq_min']
        if nfreq != 1 or abs(rhs_frequency - frequency) > 1e-3 * frequency:
            raise ConfigurationError(
                f"OASS: the mean field wrote {nfreq} frequency block(s) at "
                f"{rhs_frequency:g} Hz but OASS runs at {frequency:g} Hz. "
                f"OASS pins nfreq=1 (unoass21.f:123) and replaces its own "
                f"frequency with the .rhs's (:135) after a stdout-only "
                f"warning, so the result would carry the wrong frequency.",
                remediation=("Give the mean-field model the same single-"
                             "frequency Source this run uses."),
            )
        return mean_result, stem

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """Run the mean-field producer, then OASS.

        Parameters
        ----------
        run_mode : RunMode, optional
            ``RunMode.REVERBERATION`` (default) → a real dB :class:`Field`
            tagged ``kind='reverberation'``, shaped
            ``(n_depths, n_ranges)``. ``RunMode.COVARIANCE`` →
            :class:`Covariance` shaped ``(1, n_rcv, n_rcv)``. The convenience
            method ``compute_covariance(...)`` routes to the second.

        Returns
        -------
        Field or Covariance
        """
        self._require_run_triple(env, source, receiver)
        run_mode = self._resolve_run_mode(run_mode)
        self._reject_unsupported_run_kwargs(
            frequencies=frequencies, source_waveform=source_waveform,
            sample_rate=sample_rate, output_duration=output_duration)

        options = self._resolve_options(run_mode)
        self._reject_unreadable_options(options, run_mode)
        frequency = self._require_single_frequency(source)

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        interface = self._resolve_interface(env)
        self._warn_on_c_low_mismatch(env)

        fm = self._setup_file_manager()
        try:
            mean_result, rhs_stem = self._run_mean_field(
                env, source, receiver, fm, frequency)

            base_name = 'oass_run'
            input_file = fm.get_path(f'{base_name}.dat')
            self._log(f"Writing OASS input file: {input_file} "
                      f"(options={options})")
            write_oass_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                options=options,
                interface=interface,
                correlation_length=self.correlation_length,
                spectral_exponent=self.spectral_exponent,
                rms_roughness=self.rms_roughness,
                c_low=self.c_low if self.c_low is not None
                else self._mean_field_c_low(env),
                c_high=self.c_high,
                receiver_gains=self.receiver_gains,
            )

            # FOR045 points oass2 at the mean-field run's .rhs kernel,
            # named after that run's stem (see OASES._execute).
            proc = self._execute(base_name, fm.work_dir,
                                 extra_env={'FOR045': f'{rhs_stem}.045'})

            # Result-metadata extras shared by both products; each reader
            # below adds its own and routes the lot through _result_kwargs.
            extras = dict(
                frequencies=frequency,
                interface=interface,
                mean_field_result=mean_result,
            )
            nwvno = _oass_wavenumber_echo(proc)
            if nwvno is not None:
                extras['n_wavenumbers'] = nwvno
                # unoass21.f:209 re-derives NWVNO inside IF(REVERB) and the
                # MIN0(NWVNO,NP) clamp at :225 sits in the ELSE branch only,
                # so a reverberation run can integrate past the arrays and
                # write a full-size garbage table at exit 0. The echo above
                # is the count the binary actually used — bound it here.
                if nwvno > _OASES_NP:
                    raise ConfigurationError(
                        f"OASS sampled {nwvno} wavenumbers, past OASES' "
                        f"array bound NP = {_OASES_NP} "
                        f"(oases/src/compar.f:37-38); the reverberation "
                        f"branch has no clamp (unoass21.f:209 vs :225), so "
                        f"the table it wrote is not the scattered field.",
                        remediation="Raise c_low (WKMAX = 2π·f/c_low sets "
                                    "the count), lower the frequency, or "
                                    "shorten receiver.ranges.",
                    )

            if run_mode == RunMode.COVARIANCE:
                result = self._read_covariance(fm, base_name, receiver, proc,
                                               source, extras)
            else:
                result = self._read_reverberation(fm, base_name, receiver,
                                                  proc, source, extras)

            # The .rhs is named after the producer's stem, so it cannot go
            # through _attach_output_paths — but it follows the same rule:
            # paths only when the work dir survives (DOCUMENTATION.md §8).
            if not self.cleanup:
                rhs = fm.work_dir / f'{rhs_stem}.045'
                if rhs.exists():
                    result.metadata['rhs_file'] = str(rhs)

            self._log("OASS simulation complete")
            return result

        finally:
            fm.finish()

    def _read_reverberation(self, fm, base_name: str, receiver: Receiver,
                            proc, source: Source, extras: dict) -> Field:
        """Build the reverberation :class:`Field` from the ``.plt`` curves.

        Option ``'r'`` calls ``PLTLOS`` (``unoass21.f:402``), the routine OAST
        uses for TL, and ``oasfun22.f:322`` tags the curves ``…TLRAN`` — so
        ``read_oast_tl`` reads them unchanged. What they carry is
        ``-10·log10 E[|p_scat|²]`` (``oassun26.f:876-880``), a reverberation
        level, which is why the Field is tagged ``kind='reverberation'``
        rather than left as pressure in dB.
        """
        plt_path = self._require_output(
            [fm.get_path(f'{base_name}.plt')],
            what='a reverberation table (FOR020 .plt)', process=proc,
        )
        self._log(f"Reading OASS output: {plt_path}")
        oass_out = read_oast_tl(
            filepath=plt_path, receiver_depths=receiver.depths,
        )
        data = oass_out['tl']
        native_depths = oass_out['depths']
        native_ranges = oass_out['ranges']
        kw = self._result_kwargs(
            source,
            backend='oass',
            kind='reverberation',
            oass_quantity='reverberation_loss_db',
            **extras,
        )
        native = Field(
            data=data,
            coords={'depth': native_depths, 'range': native_ranges},
            **kw,
        )

        # G13 — Block VIII is RMIN RMAX NR and the binary rebuilds the axis as
        # R0 + (i-1)*RSTEP (unoass21.f:267-271), so a non-equispaced
        # receiver.ranges is replaced by a linspace over the same span. Same
        # treatment as OAST's native-FFT-grid mismatch.
        receiver_ranges = np.atleast_1d(np.asarray(receiver.ranges,
                                                   dtype=float))
        receiver_depths = np.atleast_1d(np.asarray(receiver.depths,
                                                   dtype=float))
        if (len(native_ranges) == len(receiver_ranges)
                and np.allclose(native_ranges, receiver_ranges)):
            result = native
        else:
            warnings.warn(
                "OASS: receiver.ranges is not the equispaced axis OASS "
                "integrates on (Block VIII is RMIN RMAX NR, rebuilt as "
                "R0+(i-1)*RSTEP at unoass21.f:267-271), so the level is "
                "linearly interpolated IN dB onto your ranges. Pass "
                "equispaced receiver.ranges to avoid the interpolation.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            result = native.resample_to(ranges=receiver_ranges,
                                        depths=receiver_depths)
            result.metadata['oass_native_ranges'] = native_ranges
            result.metadata['interpolated'] = True
        result = _mask_zero_range(result, source, 'OASS')
        self._attach_output_paths(
            result, fm.work_dir, base_name,
            primary_files=(('plt_file', '.plt'),),
        )
        return result

    def _read_covariance(self, fm, base_name: str, receiver: Receiver,
                         proc, source: Source, extras: dict) -> Covariance:
        """Build the :class:`Covariance` from the ``.xsm``.

        ``PUTXSM`` (``oasmun21_bin.f:335-346``) is the writer OASN uses, so
        ``read_oasn_covariance`` reads it unchanged — verified against the
        binary's own normalised ASCII dump on unit 24 to 6 decimals.
        FOR016 is always set, so the matrix can only appear under ``.xsm``.
        """
        cov_path = self._require_output(
            [fm.get_path(f'{base_name}.xsm')], what='a covariance file',
            process=proc,
        )
        self._log(f"Reading OASS covariance file: {cov_path}")
        cov_data = read_oasn_covariance(cov_path)

        rcv_pos = None
        rcv_depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
        if rcv_depths.size == cov_data['covariance'].shape[1]:
            rcv_pos = np.zeros((rcv_depths.size, 3), dtype=float)
            rcv_pos[:, 2] = rcv_depths
        kw = self._result_kwargs(
            source,
            backend='oass',
            n_receivers=cov_data['n_receivers'],
            title=cov_data.get('title', ''),
            **extras,
        )
        result = Covariance(
            covariance=cov_data['covariance'],
            receiver_positions=rcv_pos,
            **kw,
        )
        self._attach_output_paths(
            result, fm.work_dir, base_name,
            primary_files=(('xsm_file', '.xsm'), ('cor_file', '.cor')),
        )
        return result

    # Mirrors third_party/oases/bin/oass (oass.tex:438-448): covariance on
    # unit 16, the normalised-correlation ASCII dump on 24 (oassun26.f:1068),
    # and the unit-26 echo of the array INPRCV opens at oasnun22.f:37.
    # FOR045 is set per-run in _execute — it names the mean-field deck's stem,
    # not this one's.
    _FOR_FILES = {
        'FOR016': 'xsm',
        'FOR024': 'cor',
        'FOR026': 'chk',
    }
    # Every unit above is env-assigned, so the outputs always land on the
    # base_name suffixes and no bare fort.NN is written under this stem
    # (the mean-field OAST's fort.46 belongs to that producer's own list).
    _OUTPUT_SUFFIXES = ('.xsm', '.cor', '.plt', '.plp', '.chk')
