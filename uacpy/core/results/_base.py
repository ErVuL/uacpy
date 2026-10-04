"""Result base class, phase-reference enum and the metadata registries shared
by every result type."""

from __future__ import annotations

import contextvars
from enum import Enum
import operator
import warnings
from types import MappingProxyType

import numpy as np
from typing import Optional, Dict, Any, Mapping, Tuple, Union

from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._plotting import plotter
from uacpy.core._carrier import DeepCopyMixin
from uacpy.core._repr import axis, build, plural
from uacpy.core.results.quantities import coordinate_unit
from uacpy.core._export import (Exportable, _resolve_class, decode_attrs,
                                write_netcdf_groups, _from_json, _to_json,
                                encode_attrs, saved_class_path)
from uacpy.core.run_settings import RunMode, RunSettings


def _integer_index(value, where: str) -> int:
    """``value`` as a positional index, refusing anything that is not an
    integer: ``int(1.7)`` would select sample 1 of an axis without a word."""
    try:
        return operator.index(value)
    except TypeError:
        raise ConfigurationError(
            f"{where}={value!r} is not an integer index; isel takes a "
            f"position. Select by coordinate label with .at() instead."
        ) from None


def _count(n, who: str) -> int:
    """``n`` as a count of items to keep, refusing what a slice ``[:n]``
    would misread: a negative ``n`` slices from the END (``[:-1]`` keeps all
    but the last) and a float is truncated, both without a word. The rule
    every ``first n`` / ``top n`` selection on a result applies."""
    try:
        n = operator.index(n)
    except TypeError:
        raise ConfigurationError(
            f"{who}({n!r}): the count must be an integer.") from None
    if n < 1:
        raise ConfigurationError(
            f"{who}({n}): need n >= 1 — a negative n would slice from the "
            f"end and silently return a different subset than requested.")
    return n


def _window_pair(who: str, name: str, pair):
    """``(lo, hi)`` of a window bound ``name=pair``, either end ``None`` to
    leave it open, refusing what is not a pair and an inverted pair. The
    rule every ``window`` on a result applies."""
    try:
        low, high = pair
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"{who}: {name}={pair!r} is not a (lo, hi) pair.",
            remediation="Pass two bounds, either of which may be None "
                        "to leave that end where it is.") from None
    if low is not None and high is not None and float(low) > float(high):
        raise ConfigurationError(
            f"{who}: {name}=({low}, {high}) is inverted.",
            remediation="Give the bounds low end first.")
    return (None if low is None else float(low),
            None if high is None else float(high))


#: Set while :meth:`Result.to_netcdf` gathers a file's parts: the
#: components go into groups, so :meth:`Result._identity_attrs` does not
#: warn that one xarray object leaves them out.
_WRITING_GROUPS = contextvars.ContextVar('uacpy_writing_netcdf_groups',
                                         default=False)


class PhaseReference(str, Enum):
    """Phase convention of a complex transfer function ``H(f)`` or of a
    reflection coefficient ``R(θ)``'s phase.

    Every uacpy wrapper normalises its native phase convention before
    storing data on a broadband :class:`Field`; downstream consumers
    (IFFT, time-series synthesis) only need to know whether the payload
    is in the engineering travelling-wave form or whether it lives in
    the time domain already. Inherits from ``str`` so
    ``ref == 'travelling_wave'`` works directly.

    Members
    -------
    TRAVELLING_WAVE
        ``H(f)`` carries the engineering propagator ``exp(-i k0 r)``;
        ``2*Re[ifft(H)]`` lands the causal arrival at ``t = r/c0``.
        Used by Bellhop, Scooter, OASES OAST/OASP, Kraken, and RAM
        (mpiramS / Collins backends bake the carrier into the data). Also
        the sign of :class:`~uacpy.core.results.ReflectionCoefficient`'s
        ``phi`` (Bounce, OASR): positive below the critical angle on a
        lossy fluid half-space.
    TIME_DOMAIN_NATIVE
        The payload is ``p(t)``, or is the transform of one, so there is
        no travelling-wave carrier left to interpret. Two producers tag
        it: SPARC, whose ``H(f)`` is the FFT of an already-time-domain
        trace — consumers wanting a time series should read the
        ``RunMode.TIME_SERIES`` :class:`Field` rather than IFFT it back —
        and the synthesis helpers, which stamp it on every trace they
        build whatever convention the source ``H(f)`` carried.
    """
    TRAVELLING_WAVE = 'travelling_wave'
    TIME_DOMAIN_NATIVE = 'time_domain_native'


# ─────────────────────────────────────────────────────────────────────────────
# Documented metadata-key registry
# ─────────────────────────────────────────────────────────────────────────────

# Per ``(model_name, key)``: ``(expected_type, one-line description)``.
# ``Result.list_metadata()`` consults this so users get the type + meaning
# for every documented entry on ``result.metadata`` without grepping the
# source. Keep this synchronised with what each wrapper actually attaches
# (grep ``_attach_output_paths`` and ``_result_kwargs(`` calls per model).
_UNIVERSAL_METADATA: Dict[str, Tuple[type, str]] = {
    'work_dir': (
        str, 'The directory holding the run\'s files (only when its '
        'scratch survives, i.e. cleanup=False — the default whenever '
        'work_dir is pinned); every *_file key points inside it.'
    ),
    'prt_file': (
        str, "Acoustics-Toolbox diagnostic .prt log (only when the run's "
        'scratch survives, i.e. cleanup=False — which is the default '
        'whenever work_dir is pinned). The RAM and OASES families write no '
        'print file.'
    ),
    'run_settings_summary': (
        str, 'The one-line RunSettings.summary() of the producing run, as '
        'to_xarray writes it beside the full record (the JSON attribute '
        'run_settings, read back as result.run_settings); a result read back '
        'by from_xarray carries the line here.'
    ),
    # Attached by the shared Field synthesis helpers (``to_time_trace`` /
    # ``synthesize_time_series``), so they are model-independent.
    'source_model': (
        str, 'Model that produced the H(f) the time-domain trace was '
        'synthesised from.'
    ),
    'source_waveform_sample_rate': (
        float, 'Sample rate (Hz) of the source waveform passed to '
        'Field.synthesize_time_series.'
    ),
}

_DOCUMENTED_METADATA: Dict[Tuple[str, str], Tuple[type, str]] = {
    # ───────── Bellhop ─────────
    ('Bellhop', 'shd_file'): (str, 'Bellhop pressure-field output (.shd).'),
    ('Bellhop', 'arr_file'): (str, 'Bellhop arrivals output (.arr).'),
    ('Bellhop', 'ray_file'): (str, 'Bellhop ray paths (.ray).'),
    ('Bellhop', 'center_frequency'): (
        float, 'Carrier (centre) frequency fc (Hz) used to build H(f) '
        'from a single arrivals run.'
    ),
    ('Bellhop', 'dt'): (
        float, 'Time-sample step (s) for TIME_SERIES results '
        '(= 1 / sample_rate).'
    ),
    ('Bellhop', 'fs'): (
        float, 'Sample rate (Hz) for TIME_SERIES results.'
    ),
    ('Bellhop', 'nt'): (
        int, 'Number of samples in the TIME_SERIES time axis.'
    ),
    ('Bellhop', 't_start'): (
        float, 'Start time (s) of the delay-and-sum window.'
    ),
    # ───────── Kraken (modes + field pipeline, backend=kraken/krakenc) ─────────
    ('Kraken', 'mod_file'): (str, 'Kraken modes file (.mod).'),
    ('Kraken', 'n_modes_requested'): (
        int, 'User-supplied mode-count cap. Kraken itself does not cap '
        'mode count; this records what the wrapper sliced via .first_n().'
    ),
    ('Kraken', 'leaky_modes'): (
        bool, 'True when the run was configured to include leaky modes '
        '(c_high pushed to ~1e9). The real-arithmetic backend raises on '
        'leaky modes — they require backend=krakenc (complex k).'
    ),
    ('Kraken', 'shd_file'): (
        str, 'Kraken pressure-field output (.shd) from the field.exe step.'
    ),
    ('Kraken', 'field_prt_file'): (
        str, "field.exe's own diagnostic log — field.f90 hard-codes the "
        "name field.prt, so it is distinct from the modes run's "
        '<base_name>.prt recorded as prt_file. Attached only when '
        'cleanup=False and the log exists.'
    ),
    ('Kraken', 'mode_coupling'): (
        str, "'adiabatic', 'coupled', or 'none' (range-independent run)."
    ),
    ('Kraken', 'n_profiles'): (
        int, 'Number of modal segments used for the range-dependent '
        'field path.'
    ),
    ('Kraken', 'native_broadband'): (
        bool, 'True when Kraken produced H(f) natively from a '
        'multi-frequency .mod file (versus a Python frequency loop).'
    ),
    ('Scooter', 'grn_file'): (
        str, "Scooter Green's-function output (.grn)."
    ),
    ('Scooter', 'transform_method'): (
        str, "Hankel-transform method used by GreensFunction.to_field to "
        "map k-domain G(k) → r-domain p(r): 'direct_dft', the "
        "trapezoidal-rule matrix-product DFT (there is no FFT on this path)."
    ),
    ('Scooter', 'source_type'): (
        str, "Scooter source type passed to GreensFunction.to_field ('R' = "
        "point/cylindrical, 'X' = line/Cartesian, 'S' = point with "
        "cylindrical spreading removed)."
    ),
    ('Scooter', 'spectrum'): (
        str,
        "Wavenumber-branch selector the Hankel transform consumed: 'P' "
        "positive branch, 'N' negative branch, 'B' both."
    ),
    ('Scooter', 'center_frequency'): (
        float, 'Centre frequency (Hz) of the broadband sweep — picked '
        'as the middle element of the GreensFunction frequencies.'
    ),
    ('Scooter', 'nfreq'): (
        int, 'Number of frequencies in the broadband sweep.'
    ),
    # ───────── SPARC (time-domain FFP) ─────────
    ('SPARC', 'grn_file'): (
        str, "SPARC Green's-function snapshot (.grn, output_mode='S')."
    ),
    ('SPARC', 'rts_file'): (
        str, "SPARC received-time-series output (.rts, output_mode='R'/'D')."
    ),
    ('SPARC', 'n_depth_runs'): (
        int, "Number of per-depth SPARC subprocess calls dispatched "
        "(R-mode loops over receiver depths)."
    ),
    ('SPARC', 'n_range_runs'): (
        int, "Number of per-range SPARC subprocess calls dispatched "
        "(D-mode loops over receiver ranges)."
    ),
    ('SPARC', 'dt'): (float, 'Time-sample step (s) for TIME_SERIES output.'),
    ('SPARC', 'fs'): (float, 'Sample rate (Hz) for TIME_SERIES output.'),
    ('SPARC', 'nt'): (int, 'Number of samples in the TIME_SERIES time axis.'),
    ('SPARC', 't_start'): (
        float, 'Start time (s) of the SPARC TIME_SERIES window.'
    ),
    # Snapshot-mode SPARC attaches the GreensFunction transform keys when
    # the snapshot path computes p(z, r) via time-FFT + Hankel transform.
    # The ``snapshot_*`` / ``normalize`` / ``absolute_tl_calibrated`` keys
    # come from :meth:`uacpy.core.results.GreensFunction.snapshot_to_field`,
    # which users call directly on a .grn (the SPARC wrapper itself runs the
    # time-evolving path); stamp ``result.model = 'SPARC'`` for
    # ``list_metadata()`` to resolve them.
    ('SPARC', 'transform_method'): (
        str, "Hankel-transform method used to convert the k-domain "
        "snapshot to r-domain pressure: 'hankel_per_snapshot_time' for "
        "the time-evolving snapshot, 'time_fft+hankel' for the "
        "single-frequency steady-state snapshot."
    ),
    ('SPARC', 'source_type'): (
        str, "Source type ('R' / 'X' / 'S') consumed by the Hankel "
        "transform."
    ),
    ('SPARC', 'spectrum'): (
        str,
        "Wavenumber-branch selector the Hankel transform consumed: 'P' "
        "positive branch, 'N' negative branch, 'B' both."
    ),
    ('SPARC', 'snapshot_freq_bin'): (
        float, "Nearest FFT-bin frequency (Hz) to the source frequency; "
        "the field itself is evaluated at the source frequency."
    ),
    ('SPARC', 'snapshot_dt'): (
        float, "Time-step (s) used in the snapshot FFT."
    ),
    ('SPARC', 'snapshot_nt'): (
        int, "Number of time samples in the snapshot FFT."
    ),
    ('SPARC', 'normalize'): (
        str, "Source-spectrum deconvolution applied to the snapshot: "
        "'source' (divide by the spectrum of the source_waveform passed) or "
        "'none' (raw field)."
    ),
    ('SPARC', 'absolute_tl_calibrated'): (
        bool, "True when the snapshot was deconvolved by the source "
        "spectrum, so its TL is on the same absolute scale as "
        "Scooter / Kraken."
    ),
    # ───────── Bounce → ReflectionCoefficient ─────────
    ('Bounce', 'brc_file'): (
        str, '.brc bottom-reflection-coefficient file written by Bounce.'
    ),
    ('Bounce', 'irc_file'): (
        str, '.irc internal-reflection-coefficient file written by Bounce.'
    ),
    ('Bounce', 'n_points'): (
        int, 'Number of angle samples in the reflection-coefficient table.'
    ),
    ('RAM', 'tl_grid_file'): (str, 'tl.grid TL output from Collins backends.'),
    ('RAM', 'tl_line_file'): (
        str, "tl.line, the Collins backends' ASCII ``range  TL`` trace at the "
        'single deck receiver depth. Written every MARCH step, not every '
        'ndr-th one like tl.grid (``ramgeo1.5.f:420-425``: the write sits '
        'above the ``if(mdr.eq.ndr)`` block), so it is a finer range axis '
        'than the Field. uacpy builds its Field from tl.grid; read this with '
        'uacpy.io.read_tl_line.'
    ),
    ('RAM', 'pcomplex_file'): (str, 'pcomplex.bin complex-pressure output.'),
    ('RAM', 'in_file'): (str, 'ram.in input file consumed by the backend.'),
    ('RAM', 'psif_file'): (str, 'psif.dat broadband output from mpiramS.'),
    # ───────── OAST (TL via wavenumber integration) ─────────
    ('OAST', 'plt_file'): (str, 'OAST .plt output.'),
    ('OAST', 'oast_grid_shape'): (
        tuple,
        '(n_depths, n_ranges) of the native OAST output grid before any '
        'interpolation onto the user-supplied receiver grid.'
    ),
    ('OAST', 'native_ranges'): (
        'ndarray',
        'Native OAST range axis (m) — present when the wrapper resampled '
        'OAST onto the user receiver grid.'
    ),
    ('OAST', 'interpolated'): (
        bool,
        'True when the returned field was resampled onto the user '
        'receiver grid; False / absent when the native grid was kept.'
    ),
    ('OAST', 'plotted_frequencies'): (
        object,
        'Frequencies the .plp curves were plotted at, read back from their '
        'Freq: labels (one entry per frequency block).'
    ),
    ('OAST', 'n_frequencies'): (
        int,
        'Number of frequency blocks in the .plt file (multi-frequency '
        'decks stack NFREQ curve sets; the model path always writes 1).'
    ),
    # ───────── OASN (covariance / replicas) ─────────
    ('OASN', 'xsm_file'): (
        str, 'Covariance output (.xsm) — RunMode.COVARIANCE.'
    ),
    ('OASN', 'rpo_file'): (
        str, 'Replica output (.rpo) — RunMode.REPLICA.'
    ),
    ('OASN', 'n_receivers'): (
        int, 'Number of receivers (NRCV) in the OASN array.'
    ),
    ('OASN', 'title'): (
        str, 'Title string from the OASN output file header.'
    ),
    ('OASN', 'surface_noise_level'): (
        float, 'Surface noise level (dB re 1 µPa²/Hz) from the .xsm header '
               '(RunMode.COVARIANCE).'
    ),
    ('OASN', 'white_noise_level'): (
        float, 'White (sensor) noise level (dB re 1 µPa²/Hz) from the .xsm '
               'header; -200 when none was configured (RunMode.COVARIANCE).'
    ),
    # ───────── OASR (reflection coefficients) ─────────
    ('OASR', 'trc_file'): (
        str, 'Reflection-coefficient table (.trc).'
    ),
    ('OASR', 'rco_file'): (
        str, 'Complex reflection-coefficient output (.rco).'
    ),
    ('OASR', 'sampling_type'): (
        str, "How the angle/slowness axis was sampled by OASR "
        "('angle' or 'slowness')."
    ),
    ('OASR', 'angle_type'): (
        str, "How the requested angles were read ('grazing' or 'incidence'); "
        "``theta`` on the result is always grazing."
    ),
    ('OASR', 'reflection_type'): (
        str, "Which reflection coefficient OASR returned: 'P-P' (default), "
        "'P-SV', 'P-Slow' (Biot only), or 'transmission'."
    ),
    # ───────── OASP (pulse / broadband transfer function) ─────────
    ('OASP', 'trf_file'): (
        str, 'Transfer-function output (.trf).'
    ),
    ('OASP', 'center_frequency'): (
        float, 'OASP carrier (centre) frequency (Hz).'
    ),
    ('OASP', 'frequencies_available'): (
        'ndarray',
        'Full frequency axis available in the .trf, kept on a '
        'single-frequency slice result so the caller can recover the '
        'broadband context.'
    ),
    ('OASP', 'source_depth'): (
        float, 'Source depth (m) read from the .trf header.'
    ),
    # ───────── OASS (reverberation from a rough interface) ─────────
    ('OASS', 'plt_file'): (
        str, 'Reverberation-vs-range curve data (.plt).'
    ),
    ('OASS', 'xsm_file'): (
        str, 'Reverberation covariance (.xsm) — RunMode.COVARIANCE.'
    ),
    ('OASS', 'cor_file'): (
        str, 'Normalised spatial correlation the binary dumps as ASCII on '
        'unit 24 (oassun26.f:1068), useful as a cross-check on the .xsm.'
    ),
    ('OASS', 'rhs_file'): (
        str, 'Mean-field boundary operators consumed as FOR045 (.045), '
        'written by the OAST/OASR producer run with option "s".'
    ),
    ('OASS', 'oass_quantity'): (
        str, "Long name of the quantity on the Field: "
        "'reverberation_loss_dB'."
    ),
    ('OASS', 'n_integrated_wavenumbers'): (
        int, 'Wavenumber count OASS derived from the .rhs sampling '
        '(unoass21.f:209-215), which overrides the deck value.'
    ),
    ('OASS', 'native_ranges'): (
        'ndarray',
        'Equispaced range axis (m) OASS integrated on — present when the '
        'wrapper resampled onto a non-equispaced receiver.ranges.'
    ),
    ('OASS', 'interpolated'): (
        bool, 'True when the reverberation loss was interpolated onto the '
        'user receiver grid; False / absent when the native grid was kept.'
    ),
    ('OASS', 'n_receivers'): (
        int, 'Number of receivers (NRCV) in the OASS array.'
    ),
    ('OASS', 'title'): (
        str, 'Title string from the .xsm header. OASS leaves the COMMON the '
        'writer reads unfilled (unoass21.f:34 declares TITLE locally), so it '
        'is empty.'
    ),
    # ───────── OASSP (scattered-field realizations) ─────────
    ('OASSP', 'trf_file'): (
        str, 'Scattered-field transfer functions (.trf), in OASP format.'
    ),
    ('OASSP', 'rhs_file'): (
        str, 'Mean-field boundary operators consumed as FOR045 (.045), '
        'written by the OASP producer run with option "s".'
    ),
    ('OASSP', 'vol_file'): (
        str, 'Mean field inside the scattering layer, consumed as FOR046 '
        '(.046); OASSP opens it unconditionally (unoassp30.f:128).'
    ),
    ('OASSP', 'center_frequency'): (
        float, 'Carrier (centre) frequency (Hz) read from the .trf header.'
    ),
    ('OASSP', 'freq_max'): (
        float, 'Upper band edge FR2, from the same .rhs record as NT.'
    ),
    ('OASSP', 'source_depth'): (
        float, 'Source depth (m) read from the .trf header — the realization '
        'is returned in OASP\'s .trf format, so it carries OASP\'s header '
        'fields.'
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────


def coordinate_axis(name: str, values) -> str:
    """The coordinate ``name`` as a repr axis, in its unit: ``'12 depths
    5–95 m'``, ``'receiver depth 50 m'``."""
    word = name.replace('_', ' ')
    return axis(values, plural(word), coordinate_unit(name), singular=word)


#: A value matches an axis entry within this fraction of the axis span, the
#: value or 1, whichever is largest (:func:`axis_match_tolerance`).
_AXIS_MATCH_RTOL = 1e-6


def axis_match_tolerance(axis_values, value) -> float:
    """How far ``value`` may sit from a point of ``axis_values`` and still
    be that point: a coordinate written to an engine's deck and read back
    from its output file carries the deck's rounding."""
    axis_values = np.asarray(axis_values, dtype=float).ravel()
    return _AXIS_MATCH_RTOL * max(float(np.ptp(axis_values)),
                                  abs(float(value)), 1.0)


class Result(DeepCopyMixin, Exportable):
    """Common base for every model output.

    Carries identification (``model``, ``backend``), the source context
    (``source_depths``, ``frequencies``), and a free-form ``metadata``
    dict for model-specific extras. Subclasses add the shape-specific
    payload and methods.

    A result carries **no carriers**: never an
    :class:`~uacpy.core.environment.Environment`,
    :class:`~uacpy.core.source.Source` or
    :class:`~uacpy.core.receiver.Receiver`, and none may be added. Only the
    scalar/array identification above crosses over, so a result stays a
    self-contained record of what the model returned and the geometry keeps
    a single source of truth — the carrier the caller still holds.
    Everything that needs the geometry takes it explicitly, plotters
    included (``plot_result(result, env=…)``).

    Parameters
    ----------
    model : str
        Name of the wrapper class that produced this result (e.g. ``'RAM'``,
        ``'Bellhop'``, ``'Kraken'``).
    backend : str, optional
        The engine that actually ran, as the wrapper resolved it (e.g.
        ``'mpirams'``, ``'kraken'``, ``'krakenc'``, ``'cuda'``). Defaults to
        ``model.lower()`` when the wrapper is not a dispatcher.

        It is the *resolved* engine, which need not be the one a
        ``backend=`` argument asked for, and the relation differs per
        dispatcher: ``RAM(backend=…)`` and ``Kraken(backend=…)`` round-trip
        (Kraken's is the modes binary, whichever route sums the modes);
        ``Bellhop`` stamps the engine family
        it resolved (``'fortran'``, ``'cxx'``, ``'cuda'``, ``'custom'``),
        which ``backend=None`` picks among the installed binaries. Read this field, not the constructor
        argument, to know what produced the numbers.
    source_depths : array-like, optional
        Source depths used in the run (m). Stored as a 1-D ndarray.
    frequencies : array-like, optional
        Frequency vector in Hz, always stored as 1-D ndarray; length-1 for
        narrowband. ``result.f0`` is the first of them as a scalar — for
        narrowband, the only one.
    phase_reference : PhaseReference or str, optional
        The phase convention of complex data (:class:`PhaseReference`); an
        unknown value is refused. ``None`` for a result with no phase to
        reference (TL in dB, rays, arrivals).
    model_source : ModelProvenance, optional
        Authorship, licence and citation of the engine that ran
        (:class:`~uacpy.models.provenance.ModelProvenance`), stamped by the
        model. ``None`` when the engine declares none.
    run_mode : RunMode or str, optional
        The :class:`~uacpy.core.run_settings.RunMode` of the model run this
        result came from, stamped by ``PropagationModel.run`` on every result
        it returns and carried onto every result derived from it. It says what
        the numbers are regardless of storage: an OAST ``COHERENT_TL`` field
        stores real dB and is still a coherent field. ``RunMode`` members are
        ``str``, so a value read back from a file as ``'coherent_tl'``
        compares equal to ``RunMode.COHERENT_TL``. ``None`` for a result not
        produced by a model run.
    source_level_dB : float, optional
        The drive level of the ``Source`` that produced this result, in dB re
        1 µPa at 1 m (``Source.source_level_dB``), so
        :meth:`Field.at_source_level` needs no argument. ``None`` when the
        source stated none.
    source_weights : ndarray, optional
        The ``Source`` weights of a multi-depth run, one per source depth,
        recorded on every slab of its stack so
        :meth:`ResultStack.superpose` applies them by default. ``None`` on a
        result no weighted stack produced.
    metadata : dict, optional
        Model-specific extras (Q, T, dr, dz, n_modes, …). It never carries
        an identity field: a ``metadata`` holding ``source_level_dB`` or
        ``source_weights`` is refused.
    run_settings : RunSettings, optional
        The settings the run that produced this result actually used
        (:class:`uacpy.core.run_settings.RunSettings`), stamped by
        ``PropagationModel.run`` and carried onto every result derived from
        it; read it back as the read-only :attr:`run_settings`. ``None`` for
        a result not produced by a model run.
    components : mapping, optional
        The results this one was built from that are results in their own
        right, by name (:data:`COMPONENT_NAMES`): ``'arrivals'``, the
        :class:`Arrivals` a broadband or time-series field was synthesised
        from; ``'bounce'``, the :class:`ReflectionCoefficient` table a
        routed seabed used; ``'mean_field'``, the OASES mean field a
        reverberation run scattered. Read back as the read-only
        :attr:`components`; a derived result does not inherit them, except
        through :meth:`Field.replace` on the same quantity.
    """

    #: The source's identity fields a result carries as attributes.
    _SOURCE_IDENTITY = ('source_level_dB', 'source_weights')

    def __init__(
        self,
        *,
        model: str = "",
        backend: Optional[str] = None,
        source_depths: Optional[np.ndarray] = None,
        frequencies: Optional[Union[float, np.ndarray]] = None,
        phase_reference: Optional[str] = None,
        model_source: Optional[Any] = None,
        run_mode: Optional[str] = None,
        source_level_dB: Optional[float] = None,
        source_weights: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_settings: Optional[Any] = None,
        components: Optional[Mapping[str, 'Result']] = None,
    ):
        # An identity field held in metadata would be a second decider.
        carried = sorted(set(metadata or {}) & set(self._SOURCE_IDENTITY))
        if carried:
            raise ConfigurationError(
                f"{type(self).__name__}: metadata carries {carried}, which "
                f"are attributes of the result, not metadata.",
                remediation="Pass them as keywords: "
                            + ", ".join(f"{t}=..." for t in carried) + ".")
        self.model = model
        # Membership-checked for the reason phase_reference is below: a typo
        # ('cohernt_tl') compares unequal to every mode downstream, so each
        # consumer that branches on the run mode silently takes its
        # not-this-mode path.
        if run_mode is not None and (not isinstance(run_mode, str)
                                     or run_mode not in tuple(RunMode)):
            raise ConfigurationError(
                f"{type(self).__name__}: run_mode={run_mode!r} must be a "
                f"RunMode member or its string value, one of "
                f"{[m.value for m in RunMode]}.")
        self.run_mode: Optional[str] = run_mode
        self.backend = backend if backend is not None else (model.lower() if model else "")
        # Provenance of the engine that produced this result (a
        # ``uacpy.models.provenance.ModelProvenance`` or ``None``). Injected centrally
        # by ``PropagationModel._result_kwargs``; rendered on plots alongside
        # the data-source credit. Mirrors how ``env.data_sources`` carries
        # dataset provenance.
        self.model_source = model_source
        # Copy on ingest so a caller mutating their source array (e.g. the
        # ``Source.depths`` a model passed straight through) can't silently
        # corrupt this result.
        self.source_depths = (
            np.atleast_1d(np.array(source_depths, dtype=float))
            if source_depths is not None else np.array([], dtype=float)
        )
        # Plural-only rule: ``frequencies`` is always a 1-D ndarray of length
        # ≥ 1, or ``None`` for results that have no frequency axis (e.g.
        # SPARC native time-domain). Scalar input auto-wraps to length 1.
        if frequencies is not None:
            self.frequencies: Optional[np.ndarray] = np.atleast_1d(
                np.array(frequencies, dtype=float)
            )
        else:
            self.frequencies = None
        # Membership-checked but stored unchanged: the consumers compare it
        # against the plain strings (``field.py``'s IFFT guard reads
        # ``== 'time_domain_native'``), which the enum members satisfy through
        # their str base, so both a member and its value are accepted and
        # neither is rewritten into the other. A value outside the enum would
        # pass that comparison silently and defeat the guard.
        if phase_reference is not None:
            try:
                PhaseReference(phase_reference)
            except (ValueError, TypeError):
                raise ConfigurationError(
                    f"{type(self).__name__}: phase_reference="
                    f"{phase_reference!r} is not a known phase convention; "
                    f"pass one of "
                    f"{[m.value for m in PhaseReference]} (or the "
                    f"PhaseReference member). An unrecognised value reads as "
                    f"'not time_domain_native' everywhere downstream, so the "
                    f"IFFT/synthesis guards would let a time-domain payload "
                    f"through as a transfer function."
                ) from None
        self.phase_reference: Optional[str] = phase_reference
        self.source_level_dB: Optional[float] = (
            None if source_level_dB is None else float(source_level_dB))
        # Copied on ingest, as source_depths is.
        self.source_weights: Optional[np.ndarray] = (
            None if source_weights is None else np.array(source_weights))
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}
        self._run_settings = run_settings
        if components:
            unknown = sorted(set(components) - set(self.COMPONENT_NAMES))
            if unknown:
                raise ConfigurationError(
                    f"{type(self).__name__}: components {unknown} are not "
                    f"component names; the names are "
                    f"{list(self.COMPONENT_NAMES)}.")
            wrong = sorted(name for name, value in components.items()
                           if not isinstance(value, Result))
            if wrong:
                raise ConfigurationError(
                    f"{type(self).__name__}: components {wrong} are not "
                    f"results; a component is a Result (Arrivals, "
                    f"ReflectionCoefficient, Field ...).")
            self._components = dict(components)

    #: The names :attr:`components` may carry.
    COMPONENT_NAMES = ('arrivals', 'bounce', 'mean_field')

    @property
    def components(self) -> Mapping[str, 'Result']:
        """The results this one was built from, by name (see the
        ``components`` parameter), as a read-only mapping; empty when there
        are none. The values are the component objects themselves."""
        return MappingProxyType(getattr(self, '_components', {}))

    @property
    def run_settings(self):
        """The :class:`~uacpy.core.run_settings.RunSettings` the producing
        run used — what ``model.run_settings(...)`` returns for the same
        call — or ``None`` for a result no model run produced. Read-only:
        the record is what the run did, not a knob."""
        return self._run_settings

    # Convenience ------------------------------------------------------------

    @property
    def n_frequencies(self) -> int:
        return 0 if self.frequencies is None else int(len(self.frequencies))

    @property
    def f0(self) -> Optional[float]:
        """First frequency in Hz — the only one for a narrowband result, the
        first sample of a band otherwise, never its centre — or ``None`` when
        the result carries no frequencies."""
        if self.frequencies is None or len(self.frequencies) == 0:
            return None
        return float(self.frequencies[0])

    def id_kwargs(self) -> dict:
        """The identification fields as a kwargs dict, for cloning them onto a
        result derived from this one.

        The single home for the identity surface every ``Result`` carries, so
        adding a field to :meth:`__init__` reaches every derived-result spawn
        path at once. Public so downstream toolkits (e.g. :mod:`uacpy.sonar`)
        can carry provenance without hand-copying. Override a single entry with
        ``dict(self.id_kwargs(), frequencies=…)``."""
        return dict(
            model=self.model,
            backend=self.backend,
            source_depths=self.source_depths,
            frequencies=self.frequencies,
            phase_reference=self.phase_reference,
            model_source=self.model_source,
            run_mode=self.run_mode,
            source_level_dB=self.source_level_dB,
            source_weights=self.source_weights,
            metadata=dict(self.metadata),
            run_settings=self._run_settings,
        )

    # Persistence ------------------------------------------------------------

    def _identity_dict(self) -> Dict[str, Any]:
        """The identity block every result's ``to_dict`` writes: arrays
        copied, the phase reference and run mode as their plain string
        values. ``np.savez`` sizes a ``str``-subclass scalar by its value but
        fills it from ``str(member)``, so a member would be stored as
        ``'PhaseReference.'``. The components, when there are any, are
        nested under ``'components'``, each its own ``to_dict`` under its
        public class path."""
        out = {
            'model': self.model,
            'backend': self.backend,
            'source_depths': self.source_depths.copy(),
            'frequencies': (None if self.frequencies is None
                            else self.frequencies.copy()),
            'phase_reference': (None if self.phase_reference is None
                                else PhaseReference(self.phase_reference).value),
            'model_source': self.model_source,
            'run_mode': (None if self.run_mode is None
                         else str(getattr(self.run_mode, 'value',
                                          self.run_mode))),
            'source_level_dB': self.source_level_dB,
            'source_weights': (None if self.source_weights is None
                               else self.source_weights.copy()),
            'metadata': dict(self.metadata),
            # Plain types, like everything else here; from_dict rebuilds it.
            'run_settings': (None if self._run_settings is None
                             else self._run_settings.to_dict()),
        }
        if self.components:
            out['components'] = {
                name: {'__class__': saved_class_path(type(component)),
                       **component.to_dict()}
                for name, component in self.components.items()}
        return out

    @staticmethod
    def _unwrap_saved(d, payload=()) -> Dict[str, Any]:
        """``d`` with the 0-d arrays ``np.load(f, allow_pickle=True)`` hands
        back for a pickled dict, ``None`` or string unwrapped to their value.
        The ``payload`` keys are left alone: a fully reduced payload is
        itself a 0-d array."""
        return {k: (v.item() if k not in payload and isinstance(v, np.ndarray)
                    and v.ndim == 0 else v)
                for k, v in d.items()}

    @staticmethod
    def _identity_from_dict(d) -> Dict[str, Any]:
        """The constructor keywords for the identity :meth:`_identity_dict`
        wrote, the phase reference restored to its member and each component
        rebuilt by its own class's ``from_dict``. A source identity field a
        file keeps in its metadata is read from there. The run settings are
        the record's plain-type dict, or the record itself as
        :meth:`_identity_from_attrs` rebuilds it."""
        phase_reference = d.get('phase_reference')
        if phase_reference in {m.value for m in PhaseReference}:
            phase_reference = PhaseReference(phase_reference)
        run_settings = d.get('run_settings')
        if run_settings is not None and not isinstance(run_settings,
                                                       RunSettings):
            run_settings = RunSettings.from_dict(run_settings)
        metadata = dict(d.get('metadata') or {})
        source = {tag: metadata.pop(tag, d.get(tag))
                  for tag in Result._SOURCE_IDENTITY}
        components = {}
        for name, saved in dict(d.get('components') or {}).items():
            saved = dict(saved)
            klass = _resolve_class(saved.pop('__class__'), Result)
            components[name] = klass.from_dict(saved)
        return dict(
            components=components or None,
            model=d.get('model', ''),
            backend=d.get('backend'),
            source_depths=d.get('source_depths'),
            frequencies=d.get('frequencies'),
            phase_reference=phase_reference,
            model_source=d.get('model_source'),
            run_mode=d.get('run_mode'),
            **source,
            metadata=metadata or None,
            run_settings=run_settings,
        )

    #: ``attrs`` keys the xarray export writes for the identity surface;
    #: every other attr reads back into ``metadata``.
    _IDENTITY_ATTRS = ('model', 'backend', 'phase_reference', 'run_mode',
                       'frequencies', 'source_depths', 'model_source_id',
                       'run_settings')

    #: The attrs :meth:`to_netcdf` adds to each part of a file: the class
    #: that reads the part back, and the names of its components (each in
    #: the group ``components/<name>`` below the part).
    _FILE_ATTRS = ('uacpy_class', 'uacpy_components')

    def _identity_attrs(self) -> Dict[str, Any]:
        """The identity as NetCDF attributes: ``model``, ``backend``,
        ``phase_reference`` and ``run_mode`` as strings, ``frequencies`` and
        ``source_depths`` as arrays, the producing engine as
        ``model_source_id``, and the run settings as the JSON record
        ``run_settings`` (read back as :attr:`run_settings`) beside the
        one-line ``run_settings_summary`` (which reads back as metadata);
        unset entries are left out."""
        attrs: Dict[str, Any] = {'model': self.model,
                                 'backend': self.backend or ''}
        if self.phase_reference is not None:
            attrs['phase_reference'] = str(getattr(
                self.phase_reference, 'value', self.phase_reference))
        if self.run_mode is not None:
            attrs['run_mode'] = str(getattr(self.run_mode, 'value',
                                            self.run_mode))
        if self.frequencies is not None:
            attrs['frequencies'] = np.asarray(self.frequencies, dtype=float)
        if self.source_depths.size:
            attrs['source_depths'] = np.asarray(self.source_depths,
                                                dtype=float)
        source_id = getattr(self.model_source, 'id', None)
        if source_id is not None:
            attrs['model_source_id'] = str(source_id)
        if self.run_settings is not None:
            import json
            attrs['run_settings'] = json.dumps(
                _to_json(self.run_settings.to_dict()))
            attrs['run_settings_summary'] = self.run_settings.summary()
        if self.components and not _WRITING_GROUPS.get():
            warnings.warn(
                f"{type(self).__name__}: components {sorted(self.components)} "
                f"are not held by one xarray object; to_netcdf() writes each "
                f"as a NetCDF group and from_netcdf() reads them back, and "
                f"to_dict() nests them.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return attrs

    def to_netcdf(self, path, **kwargs) -> None:
        """Write this result to a NetCDF file at ``path``: its
        :meth:`to_xarray` at the file's root, and each of its
        :attr:`components` (with their own) in the group
        ``components/<name>`` below it. Each part records its class
        (``uacpy_class``) and its components' names
        (``uacpy_components``), so :meth:`from_netcdf` rebuilds the whole
        result. Each part is written as
        :func:`~uacpy.core._export.write_netcdf_groups` writes it: complex
        values as real and imaginary parts on a trailing
        ``_pfnc_complex`` dimension, boolean attributes as ``int8`` — what
        every backend stores and any netCDF reader opens.

        Parameters
        ----------
        path : str or path-like
            The file to write (replaced if it exists).
        **kwargs
            Passed to xarray's ``to_netcdf`` (an ``engine=``, for one).

        """
        parts = []
        token = _WRITING_GROUPS.set(True)
        try:
            self._netcdf_parts(None, parts)
        finally:
            _WRITING_GROUPS.reset(token)
        write_netcdf_groups(parts, path,
                            who=f"{type(self).__name__}.to_netcdf", **kwargs)

    def _netcdf_parts(self, group, parts) -> None:
        """Append ``(group, xarray object)`` for this result and, below it,
        each component."""
        obj = self.to_xarray()
        obj.attrs['uacpy_class'] = saved_class_path(type(self))
        obj.attrs['uacpy_components'] = ','.join(self.components)
        parts.append((group, obj))
        for name, component in self.components.items():
            below = f'components/{name}' if group is None \
                else f'{group}/components/{name}'
            component._netcdf_parts(below, parts)

    @classmethod
    def from_netcdf(cls, path, **kwargs) -> 'Result':
        """The result :meth:`to_netcdf` wrote to ``path``, its components
        read back from their groups.

        The class each part names (``uacpy_class``) rebuilds it, and must be
        this class (or a subclass) for the root and a :class:`Result` for a
        component. The complex values the file stored as real and
        imaginary parts are joined back
        (:func:`~uacpy.core._export.join_complex`), and the booleans it
        stored as ``int8`` (:func:`~uacpy.core._export.portable_attrs`)
        come back booleans through each class's ``from_xarray``. A file written by
        :meth:`to_xarray` and xarray alone, which names no class, is read by
        this class's :meth:`from_xarray`.

        Parameters
        ----------
        path : str or path-like
            The file to read.
        **kwargs
            Passed to ``xarray.open_dataset``. Without an ``engine``, the
            one that also reads a file holding complex values natively
            (:func:`~uacpy.core._export.complex_netcdf_backend`: h5netcdf,
            else netCDF4 1.7.1 or later with ``auto_complex=True``).

        Returns
        -------
        Result
            The result, of the class the file names, with its components.

        Raises
        ------
        ConfigurationError
            A root of another class, a class path outside uacpy, or a
            variable that comes back as netCDF4's ``(r, i)`` compound
            record because no backend that reads complex values is
            installed.
        """
        from uacpy.core._export import complex_netcdf_backend, require_extra
        xarray = require_extra('xarray', f'{cls.__name__}.from_netcdf')
        if 'engine' not in kwargs:
            backend = complex_netcdf_backend()
            if backend is not None:
                kwargs['engine'] = backend
        if kwargs.get('engine') == 'netcdf4':
            kwargs.setdefault('auto_complex', True)
        return cls._from_netcdf_group(xarray, path, None, cls, kwargs,
                                      f'{cls.__name__}.from_netcdf')

    @staticmethod
    def _from_netcdf_group(xarray, path, group, base, kwargs,
                           who) -> 'Result':
        from uacpy.core._export import (join_complex,
                                        refuse_complex_without_backend)
        with xarray.open_dataset(path, group=group, **kwargs) as ds:
            ds = join_complex(ds.load())
        # netCDF4 without auto_complex reads a complex variable as an
        # (r, i) compound record
        for name, variable in ds.variables.items():
            if variable.dtype.names is not None and \
                    set(variable.dtype.names) == {'r', 'i'}:
                refuse_complex_without_backend(who, str(name))
        if 'uacpy_class' in ds.attrs or len(ds.data_vars) != 1:
            obj = ds
        else:
            (name,) = ds.data_vars
            obj = ds[name]
            if name == '__xarray_dataarray_variable__':
                obj.name = None
        path_attr = obj.attrs.pop('uacpy_class', None)
        names = [n for n in str(obj.attrs.pop('uacpy_components', '') or '')
                 .split(',') if n]
        klass = base if path_attr is None else _resolve_class(path_attr, base)
        result = klass.from_xarray(obj)
        components = {
            name: Result._from_netcdf_group(
                xarray, path,
                f'components/{name}' if group is None
                else f'{group}/components/{name}', Result, kwargs, who)
            for name in names}
        if components:
            result._components = components
        return result

    def _export_attrs(self) -> Dict[str, Any]:
        """The identity (:meth:`_identity_attrs`) plus every metadata
        entry a NetCDF attribute can hold (:func:`encode_attrs`)."""
        attrs = self._identity_attrs()
        encode_attrs(self.metadata or {}, attrs, skip=self._IDENTITY_ATTRS,
                     who=f"{type(self).__name__}.to_xarray")
        return attrs

    @classmethod
    def _identity_from_attrs(cls, attrs: Dict[str, Any],
                             reserved=()) -> Dict[str, Any]:
        """The identity constructor keywords from the attrs
        :meth:`_export_attrs` wrote: ``metadata`` is every other attr outside
        ``reserved``, the encoded entries restored (:func:`decode_attrs`),
        with the engine recorded by id as ``metadata['model_source_id']``,
        since the engine object does not travel through a file, and the
        ``run_settings`` record rebuilt from its JSON."""
        metadata = decode_attrs(
            attrs, (*cls._IDENTITY_ATTRS, *cls._FILE_ATTRS, *reserved))
        if 'model_source_id' in attrs:
            metadata['model_source_id'] = attrs['model_source_id']
        phase_reference = attrs.get('phase_reference')
        if phase_reference in {m.value for m in PhaseReference}:
            phase_reference = PhaseReference(phase_reference)
        run_settings = None
        if attrs.get('run_settings'):
            import json
            run_settings = RunSettings.from_dict(
                _from_json(json.loads(attrs['run_settings'])))
        return dict(
            model=str(attrs.get('model', '')),
            backend=(str(attrs['backend']) if attrs.get('backend') else None),
            phase_reference=phase_reference,
            run_mode=attrs.get('run_mode'),
            frequencies=attrs.get('frequencies'),
            source_depths=attrs.get('source_depths'),
            metadata=metadata or None,
            run_settings=run_settings,
        )

    def __repr__(self) -> str:
        bits = [self.model or None]
        if self.frequencies is not None and len(self.frequencies):
            bits.append(axis(self.frequencies, 'frequencies', 'Hz'))
        return build(type(self).__name__, bits + self._repr_bits())

    def _repr_bits(self) -> list:
        """Override on subclasses to add what the result holds (``'12
        modes'``, the receiver axes, …) to :meth:`__repr__`."""
        return []

    def plot(self, **kwargs):
        """Plot this result via :func:`uacpy.plot.plot_result`.

        Dispatches on result type; concrete subclasses may override. ``kwargs``
        are forwarded to the selected plotter.
        """
        return plotter('plot_result')(self, **kwargs)

    def list_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Describe every key currently in ``self.metadata``.

        For each key, return the runtime value type, the documented
        expected type (if uacpy knows about it), and a one-line
        description. ``Bounce``'s ``'n_points'`` lookup, for example::

            ref = Bounce(work_dir=tmp).run(env, src, rcv)
            ref.list_metadata()['n_points']
            # {'value_type': 'int',
            #  'documented_type': 'int',
            #  'description': 'Number of angle samples in the '
            #                 'reflection-coefficient table.'}

        Undocumented keys still appear (with ``documented_type=None``
        and ``description=None``) so callers can see everything the
        wrapper attached.
        """
        out: Dict[str, Dict[str, Any]] = {}
        for key, value in self.metadata.items():
            doc = _DOCUMENTED_METADATA.get((self.model, key))
            if doc is None:
                doc = _UNIVERSAL_METADATA.get(key)
            if doc is not None:
                # ``documented_type`` may already be a string for
                # late-bound forward references (e.g. ``'ndarray'``,
                # ``'Arrivals'``); accept either a type or a name.
                documented_type = (
                    doc[0] if isinstance(doc[0], str) else doc[0].__name__
                )
                description = doc[1]
            else:
                documented_type = None
                description = None
            out[key] = {
                'value_type': type(value).__name__,
                'documented_type': documented_type,
                'description': description,
            }
        return out
