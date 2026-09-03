"""Result base class, phase-reference enum and the metadata registries shared
by every result type."""

from __future__ import annotations

import copy as _copy
from enum import Enum
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union

from uacpy.core.constants import PRESSURE_FLOOR
from uacpy.core.exceptions import ConfigurationError


def _complex_to_db(data: np.ndarray) -> np.ndarray:
    """``-20·log10(|data|)`` with ``|data|`` clamped to :data:`PRESSURE_FLOOR`.

    Canonical TL conversion used by :attr:`Field.db` and the metrics in
    :mod:`uacpy.core.metrics`. Preserves shape — no squeeze. The clamp caps
    an exactly-zero sample (a cell no energy reached) at 600 dB rather than
    ``+inf``, keeping the array finite for plotting and reductions.
    """
    return -20.0 * np.log10(np.maximum(np.abs(data), PRESSURE_FLOOR))


class PhaseReference(str, Enum):
    """Phase convention of a complex transfer function ``H(f)``.

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
        (mpiramS / Collins backends bake the carrier into the data).
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
    'prt_file': (
        str, 'Acoustics-Toolbox / RAM diagnostic .prt log (only when '
        'work_dir is pinned).'
    ),
    # Attached by the shared Field synthesis helpers (``to_time_trace`` /
    # ``synthesize_time_series``), so they are model-independent.
    'window': (
        str, "Band-edge taper applied to H(f) before the IFFT: 'hann', "
        "'hamming', 'blackman', 'tukey' or 'none'."
    ),
    'source_model': (
        str, 'Model that produced the H(f) the time-domain trace was '
        'synthesised from.'
    ),
    'source_waveform_sample_rate': (
        float, 'Sample rate (Hz) of the source waveform passed to '
        'Field.synthesize_time_series.'
    ),
    'c_max': (
        float, 'Fastest sound speed (m/s) in the modelled waveguide. Read by '
        'the time-series synthesis helpers to anchor the output window before '
        'the earliest arrival (r / c_max); without it the window start is an '
        'estimate and long-range traces warn.'
    ),
}

_DOCUMENTED_METADATA: Dict[Tuple[str, str], Tuple[type, str]] = {
    # ───────── Bellhop ─────────
    ('Bellhop', 'shd_file'): (str, 'Bellhop pressure-field output (.shd).'),
    ('Bellhop', 'arr_file'): (str, 'Bellhop arrivals output (.arr).'),
    ('Bellhop', 'ray_file'): (str, 'Bellhop ray paths (.ray).'),
    ('Bellhop', 'receiver_depths'): (
        np.ndarray,
        'Receiver depths (m) paired one-to-one with the range axis of an '
        'irregular-grid (RunType(5:5)=I) field, which carries no depth axis '
        'of its own.'),
    ('Bellhop', 'c0'): (
        float, 'Sea-surface sound speed (m/s) of the first profile. One of '
        'the candidate speeds the time-series synthesis helpers take the max '
        'of to anchor the output window (r / c), and the speed the range-span '
        'wrap warning measures travel time with. The arrivals → H(f) path '
        'itself needs no reference speed — the arrival delays carry the '
        'timing.'
    ),
    ('Bellhop', 'center_frequency'): (
        float, 'Carrier (centre) frequency fc (Hz) used to build H(f) '
        'from a single arrivals run.'
    ),
    ('Bellhop', 'arrivals_field'): (
        'Arrivals',
        'The intermediate Arrivals Result the broadband path was built '
        'from — kept so the caller can inspect ray fans or re-synthesise '
        'with a different waveform.'
    ),
    ('Bellhop', 'bounce_result'): (
        'ReflectionCoefficient',
        'Auto-routed Bounce result when the env carried a layered or '
        'elastic bottom and Bellhop ran against a generated .brc.'
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
    ('Kraken', 'c_low'): (
        float, 'Resolved lower phase-speed bound (m/s) of the mode search. '
        'The constructor c_low when pinned, else auto-derived — 0.0 for a '
        'fluid environment, which hands the choice to KRAKEN, and the '
        'slowest compressional speed when the bottom carries shear.'
    ),
    ('Kraken', 'c_high'): (
        float, 'Resolved upper phase-speed bound (m/s) of the mode search. '
        'The constructor c_high when pinned, else auto-derived from the SSP '
        'and bottom half-space. A range-dependent (multi-profile) run '
        'resolves it per profile and reports the widest.'
    ),
    ('Kraken', 'rmax'): (
        float, 'Resolved maximum range (m) written to the deck. KRAKEN uses it '
        'only as the mesh-convergence tolerance of the Richardson '
        'extrapolation (kraken.f90:80, `Error*1000*RMax < 1`), so a LARGER '
        'value is a TIGHTER tolerance; field.exe never reads it. The '
        'constructor rmax_m when pinned, else derived from the receiver '
        'ranges.'
    ),
    # ───────── Scooter (FFP / wavenumber integration) ─────────
    ('Scooter', 'grn_file'): (
        str, "Scooter Green's-function output (.grn)."
    ),
    ('Scooter', 'transform_method'): (
        str, "Hankel-transform method used by grn_reader to map "
        "k-domain G(k) → r-domain p(r): 'direct_dft', the trapezoidal-rule "
        "matrix-product DFT (there is no FFT on this path)."
    ),
    ('Scooter', 'source_type'): (
        str, "Scooter source type passed to grn_to_field ('R' = "
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
        'as the middle element of freqVec.'
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
    ('SPARC', 'output_mode'): (
        str, "Which SPARC output path produced this Field: 'R' "
        "(horizontal time-marched array), 'D' (vertical), 'S' (snapshot)."
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
    # Snapshot-mode SPARC attaches grn_reader keys when the snapshot
    # path computes p(z, r) via time-FFT + Hankel transform. The
    # ``snapshot_*`` / ``normalize`` / ``absolute_tl_calibrated`` keys come
    # from :func:`uacpy.io.sparc_snapshot_to_field`, which users call directly
    # on a .grn (the SPARC wrapper itself runs the time-evolving path); stamp
    # ``result.model = 'SPARC'`` for ``list_metadata()`` to resolve them.
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
        float, "FFT-bin frequency (Hz) the snapshot was extracted at "
        "(closest bin to the source frequency)."
    ),
    ('SPARC', 'snapshot_dt'): (
        float, "Time-step (s) used in the snapshot FFT."
    ),
    ('SPARC', 'snapshot_nt'): (
        int, "Number of time samples in the snapshot FFT."
    ),
    ('SPARC', 'normalize'): (
        str, "Source-spectrum deconvolution applied to the snapshot: "
        "'source' (divide by the pulse spectrum) or 'none' (raw field)."
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
    ('Bounce', 'c_low'): (
        float, 'Lower phase-speed bound (m/s) passed to Bounce.'
    ),
    ('Bounce', 'c_high'): (
        float, 'Upper phase-speed bound (m/s) passed to Bounce.'
    ),
    ('Bounce', 'rmax'): (
        float, 'Maximum range (m) passed to Bounce.'
    ),
    ('Bounce', 'n_points'): (
        int, 'Number of angle samples in the reflection-coefficient table.'
    ),
    ('Bounce', 'full_result'): (
        dict, 'Raw .brc dict (theta, R, phi, n_pts) for downstream '
        'consumers that prefer the unwrapped form.'
    ),
    # ───────── RAM dispatcher (mpiramS / rams0.5 / ramsurf1.5) ─────────
    ('RAM', 'pe_reference_speed'): (
        float, 'Padé expansion point c0 (m/s) the PE march was initialised '
        'at — an algorithmic reference, not a physical medium speed (the '
        'physical extremes ride on c_min / c_max).'
    ),
    ('RAM', 'c_min'): (
        float, 'Minimum sound speed (m/s) the solver brackets — used by '
        'the per-wavelength stability cap on dr.'
    ),
    ('RAM', 'dr'): (float, 'Range step (m) used by the PE.'),
    ('RAM', 'dz'): (float, 'Depth step (m) used by the PE.'),
    ('RAM', 'zmax'): (float, 'PE domain depth (m) including absorbing layer.'),
    ('RAM', 'Q'): (float, 'Broadband Q = fc/bandwidth (mpiramS).'),
    ('RAM', 'T'): (float, 'Time-window width (s) for broadband synthesis.'),
    ('RAM', 'bandwidth_hz'): (
        float, 'Broadband bandwidth (Hz) the run actually used.'
    ),
    ('RAM', 'df_hz'): (
        float, 'Frequency-bin spacing (Hz) of the broadband sweep.'
    ),
    ('RAM', 'n_samples'): (
        int, 'Number of time-domain samples per receiver in TIME_SERIES.'
    ),
    ('RAM', 'fs'): (
        float, 'Time-series sample rate (Hz) for TIME_SERIES results.'
    ),
    ('RAM', 'tl_grid_file'): (str, 'tl.grid TL output from Collins backends.'),
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
    ('OAST', 'oast_native_ranges'): (
        'ndarray',
        'Native OAST range axis (m) — present when the wrapper resampled '
        'OAST onto the user receiver grid.'
    ),
    ('OAST', 'interpolated'): (
        bool,
        'True when the returned field was resampled onto the user '
        'receiver grid; False / absent when the native grid was kept.'
    ),
    ('OAST', 'frequencies'): (
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
    ('OASP', 'n_time_samples'): (
        int, 'Power-of-two FFT length used by OASP for the time axis.'
    ),
    ('OASP', 'freq_max'): (
        float, 'Maximum frequency (Hz) of the OASP broadband sweep.'
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
    ('OASS', 'kind'): (
        str, "Field quantity tag: 'reverberation'. The data is "
        "-10*log10 E[|p_scat|^2] — REVINT's dB conversion at "
        "oassun26.f:853-858, where CVMAGS squares an accumulator that is "
        "already an intensity, VCLIP floors it at 1e-30, and VALG10 then "
        "VSMUL by -5E0 give the -10*log10 on that intensity. The leading "
        "minus makes it a LOSS: a larger value is a weaker scattered field, "
        "and RL = SL - this. Not transmission loss, though, so it does not "
        "compare against a TL field. REVRAN's block at oassun26.f:633-638 is "
        "byte-identical arithmetic and is NOT this: it accumulates a "
        "cross-range covariance and writes CFF(1,1), while REVINT writes "
        "CFFs, the array unoass21.f:38 equivalences to the XS that PLTLOS "
        "plots into the .plt read here. Option 'r' reaches REVINT; REVRAN "
        "belongs to the capital-'C' CCONTU contour branch "
        "(unoass21.f:626-628). OASES option letters are case-sensitive: "
        "lowercase 'c' sets ICONTU, the depth-integrand contours "
        "(unoass21.f:602-604), which reach neither routine."
    ),
    ('OASS', 'oass_quantity'): (
        str, "Long name of the quantity on the Field: "
        "'reverberation_loss_db'."
    ),
    ('OASS', 'interface'): (
        int, 'OASES deck-layer index (INTFC) of the scattering interface the '
        'reverberation was computed for.'
    ),
    ('OASS', 'n_wavenumbers'): (
        int, 'Wavenumber count OASS derived from the .rhs sampling '
        '(unoass21.f:209-215), which overrides the deck value.'
    ),
    ('OASS', 'mean_field_result'): (
        'Field', 'The mean-field Result the .rhs came from, kept so the '
        'coherent field need not be recomputed.'
    ),
    ('OASS', 'oass_native_ranges'): (
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
    ('OASSP', 'realization'): (
        int, 'Realization index k. The OASES seed is -123 - k '
        '(unoassp30.f:170, :535), so a given k is reproducible.'
    ),
    ('OASSP', 'interface'): (
        int, 'OASES deck-layer index the scattering was computed at, read '
        'from the .rhs (unoassp30.f:546-547) rather than from the deck.'
    ),
    ('OASSP', 'center_frequency'): (
        float, 'Carrier (centre) frequency (Hz) read from the .trf header.'
    ),
    ('OASSP', 'n_time_samples'): (
        int, 'FFT length NT, taken from the mean-field .rhs — OASSP replaces '
        'the deck value with it (unoassp30.f:181-188).'
    ),
    ('OASSP', 'freq_max'): (
        float, 'Upper band edge FR2, from the same .rhs record as NT.'
    ),
    ('OASSP', 'mean_field_result'): (
        'Field', 'The OASP mean-field Result the .rhs came from, kept so the '
        'coherent field need not be recomputed.'
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


class Result:
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
        Concrete binary that ran (e.g. ``'mpiramS'``, ``'kraken.exe'``,
        ``'bellhop'``). Defaults to ``model.lower()`` when the wrapper is
        not a dispatcher.
    source_depths : array-like, optional
        Source depths used in the run (m). Stored as a 1-D ndarray.
    frequencies : array-like, optional
        Frequency vector in Hz, always stored as 1-D ndarray; length-1 for
        narrowband. Use ``result.f0`` to access the centre/single frequency
        as a scalar.
    metadata : dict, optional
        Model-specific extras (Q, T, dr, dz, n_modes, …).
    """

    # Lower-case string discriminator naming the result kind, for callers and
    # printouts that want a tag. ``isinstance`` is the preferred check —
    # ``visualization.plots.plot_result`` dispatches on type, not on this.
    field_type: str = ""

    def __init__(
        self,
        *,
        model: str = "",
        backend: Optional[str] = None,
        source_depths: Optional[np.ndarray] = None,
        frequencies: Optional[Union[float, np.ndarray]] = None,
        phase_reference: Optional[str] = None,
        model_source: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.backend = backend if backend is not None else (model.lower() if model else "")
        # Provenance of the engine that produced this result (a
        # ``uacpy.models.sources.ModelSource`` or ``None``). Injected centrally
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
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}

    # Convenience ------------------------------------------------------------

    @property
    def n_frequencies(self) -> int:
        return 0 if self.frequencies is None else int(len(self.frequencies))

    @property
    def f0(self) -> Optional[float]:
        """First / centre frequency in Hz, or ``None`` for time-domain results."""
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
            metadata=dict(self.metadata),
        )

    def copy(self):
        """Deep copy of the result (symmetric with the carriers / Source /
        Receiver / Environment)."""
        return _copy.deepcopy(self)

    def __repr__(self) -> str:
        cls = type(self).__name__
        bits = [f"model={self.model!r}"] if self.model else []
        if self.frequencies is not None and len(self.frequencies):
            if len(self.frequencies) == 1:
                bits.append(f"f={float(self.frequencies[0]):.3g} Hz")
            else:
                bits.append(f"n_f={len(self.frequencies)}")
        extra = self._repr_extra()
        if extra:
            bits.append(extra)
        return f"{cls}({', '.join(bits)})"

    def _repr_extra(self) -> str:
        """Override on subclasses to add a per-class size summary
        (``n_modes=N``, ``n_rays=N``, …) to :meth:`__repr__`."""
        return ''

    def plot(self, **kwargs):
        """Plot this result via :func:`uacpy.visualization.plot_result`.

        Dispatches on result type; concrete subclasses may override. ``kwargs``
        are forwarded to the selected plotter.
        """
        # Deferred into the body: ``uacpy.visualization`` imports
        # ``uacpy.core`` at module scope, so this line at file scope makes
        # ``import uacpy`` raise ImportError. docs/DEV.md section 7 records
        # the inversion.
        from uacpy.visualization import plots
        return plots.plot_result(self, **kwargs)

    def list_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Describe every key currently in ``self.metadata``.

        For each key, return the runtime value type, the documented
        expected type (if uacpy knows about it), and a one-line
        description. ``Bounce``'s ``'c_low'`` lookup, for example::

            ref = Bounce(work_dir=tmp).run(env, src, rcv)
            ref.list_metadata()['c_low']
            # {'value_type': 'float',
            #  'documented_type': 'float',
            #  'description': 'Lower phase-speed bound (m/s) passed to Bounce.'}

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
