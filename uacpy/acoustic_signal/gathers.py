"""The gather transforms: a whole gather of traces into another domain.

:func:`fk_transform`, :func:`taup_transform` and :func:`radon_transform`,
each with an inverse. Every one takes the receiver spacing ``dx``, which is
what separates them from the single-channel estimators.
"""

from __future__ import annotations

import warnings
from collections import namedtuple
import numpy as np
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.acoustic_signal._results import PlottedResult
from uacpy.core._validate import (
    require_positive_finite_scalar, require_real_signal,
)
from uacpy.acoustic_signal.windows import (_default_noverlap, _fk_tapers,
                                           _spectral_window)


# ──────────────────────────────────────────────────────────────────────
# Gather transforms
#
# A whole panel into another domain and back:
# f-k, tau-p and Radon. Each takes the receiver spacing ``dx``, which is what
# separates them from the single-channel estimators.
# ──────────────────────────────────────────────────────────────────────

_RADON_KINDS = ("linear", "parabolic", "hyperbolic")


_RadonFields = namedtuple("RadonResult", "moveout taus panel")


class RadonResult(PlottedResult, _RadonFields):
    """Radon panel over ``moveout`` and intercept ``taus``, carrying the
    moveout family it was scanned in as ``kind``.

    The tuple is the measurement, so ``moveout, taus, panel = ...``
    keeps working; :meth:`plot` is the one obvious way to draw it. ``kind``
    is one of ``'linear'`` (slowness, s/m), ``'parabolic'`` (curvature,
    s/m²) or ``'hyperbolic'`` (velocity, m/s): it says what ``moveout``
    measures, so it is an attribute rather than a fourth tuple element, as
    ``FKResult.scaling`` is.
    """

    _attrs = ("kind",)
    _plotter = "plot_radon"

    def __new__(cls, moveout, taus, panel, *, kind):
        if kind not in _RADON_KINDS:
            raise ConfigurationError(
                f"RadonResult: kind must be one of {_RADON_KINDS}; got "
                f"{kind!r}.")
        self = super().__new__(cls, moveout, taus, panel)
        self.kind = kind
        return self

    def _field_units(self):
        moveout = {"linear": "s/m", "parabolic": "s/m²",
                   "hyperbolic": "m/s"}[self.kind]
        return {"moveout": moveout, "taus": "s", "panel": None}


class TauPResult(PlottedResult,
                 namedtuple("TauPResult", "slownesses taus panel")):
    """Tau-p panel over ``slownesses`` and intercept ``taus``.

    The tuple is the measurement, so ``slownesses, taus, panel = ...``
    keeps working; :meth:`plot` is the one obvious way to draw it.
    """

    __slots__ = ()

    _plotter = "plot_taup"
    _plot_fields = ("slownesses", "taus", "panel")

    def _field_units(self):
        return {"slownesses": "s/m", "taus": "s", "panel": None}


#: What :attr:`FKResult.scaling` may hold: ``'density'`` for the
#: calibrated panel of ``fk_transform(..., scaling='density')`` (x² per
#: Hz·rad/m, two-sided in f and k, ``ΣP·Δf·Δk = ⟨x²⟩``) and ``'power'`` for
#: the raw ``|FK|²`` of the windowed, zero-padded FFT, which carries no
#: physical unit.
FK_SCALINGS = ("density", "power")


_FKFields = namedtuple("FKResult", "frequencies wavenumbers power spectrum")


class FKResult(PlottedResult, _FKFields):
    """The ``(frequencies, wavenumbers, power, spectrum)`` 4-tuple of
    :func:`fk_transform`, carrying the panel's ``scaling`` as an attribute.

    Unpacking stays four-wide (``f, k, power, spectrum = fk_transform(...)``)
    and :func:`inverse_fk` keeps taking the fourth element. ``scaling`` is one
    of :data:`FK_SCALINGS` and tells :func:`~uacpy.plot.plot_fk`
    which unit the panel is in, so it is not a fifth tuple element: a fifth
    element would change every unpack site for one flag the plotter reads.
    """

    _attrs = ("scaling",)
    _plotter = "plot_fk"

    def __new__(cls, frequencies, wavenumbers, power, spectrum, *, scaling):
        if scaling not in FK_SCALINGS:
            raise ConfigurationError(
                f"FKResult: scaling must be one of {FK_SCALINGS}; got "
                f"{scaling!r}.")
        self = super().__new__(cls, frequencies, wavenumbers, power, spectrum)
        self.scaling = scaling
        return self

    def _field_units(self):
        # 'density' is the calibrated panel of scaling='density' (x² per
        # Hz·rad/m); 'power' is the raw |FK|², calibrated to no unit.
        power = "Pa²/(Hz·rad/m)" if self.scaling == "density" else None
        return {"frequencies": "Hz", "wavenumbers": "rad/m",
                "power": power, "spectrum": None}


def _warn_spatial_aliasing(D, freqs, dx, slownesses, who):
    """Warn when the requested slowness range outruns the trace spacing.

    A slant stack reads the moveout only at the sensors, so between adjacent
    traces it sees the phase ``2*pi*f*p*dx`` modulo a turn. Past half a turn
    the stack can no longer tell ``p`` from ``p -/+ 1/(f*dx)``: measured on a
    900 Hz plane wave at ``p = +4e-4`` s/m with ``dx = 2`` m, the panel peaks
    at ``-1.550e-4`` against ``-1.556e-4`` predicted — the right event, the
    wrong slowness, and the wrong direction of travel.

    Nothing recovers it. The wavenumber was undersampled by the array before
    the transform ran, which is why there is no spatial zero-padding knob
    here: padding a sum over sensors with zero traces adds zero terms to that
    sum and is a bit-exact no-op (unlike :func:`fk_transform`, whose spatial
    FFT length sets the wavenumber grid and so does interpolate it). The
    remedies are all upstream of the panel — a narrower slowness range, a
    low-passed gather, or a finer ``dx``.

    The bound is taken against the frequency below which 99% of the record's
    energy lies, not the Nyquist rate, so a narrowband gather is judged on the
    band it occupies. The quantile is deliberately generous: window leakage
    already lifts it (a clean 200 Hz tone reads 218.8 Hz, 9% high), which
    makes the guard fire slightly early rather than slightly late, and at
    ``p_max``'s 1e-3 default a 200 Hz tone still passes quietly.
    """
    p_max = float(np.max(np.abs(slownesses))) if slownesses.size else 0.0
    if p_max <= 0.0 or dx <= 0.0:
        return
    power = (np.abs(D) ** 2).sum(axis=1)
    total = power.sum()
    if not np.isfinite(total) or total <= 0.0:
        return              # a silent gather aliases nothing
    f_edge = float(freqs[np.searchsorted(np.cumsum(power) / total, 0.99)])
    if f_edge <= 0.0:
        return              # all energy at DC: every slowness is unaliased
    p_alias = 1.0 / (2.0 * f_edge * dx)
    if p_max <= p_alias:
        return
    # The three remedies are quoted rounded, and a reader types the quoted
    # number back. Round each one the SAFE way — down for a ceiling on |p| and
    # on dx, down for the low-pass corner — so that following the message
    # literally clears the guard instead of tripping it again on the third
    # significant figure.

    def _floor_sig(v, n=3):
        if not np.isfinite(v) or v <= 0.0:
            return v
        scale = 10.0 ** (np.floor(np.log10(v)) - (n - 1))
        return float(np.floor(v / scale) * scale)

    p_safe = _floor_sig(p_alias)
    f_safe = _floor_sig(1.0 / (2.0 * p_max * dx))
    dx_safe = _floor_sig(1.0 / (2.0 * f_edge * p_max))
    warnings.warn(
        f"{who}: the slowness axis reaches |p| = {p_max:.2e} s/m, but "
        f"with dx = {dx:g} m and 99% of the record's energy below "
        f"{f_edge:.0f} Hz the stack aliases beyond |p| = {p_safe:.3g} s/m — "
        f"an event steeper than that is indistinguishable from p -/+ "
        f"1/(f*dx) and can surface at the wrong slowness, or the wrong sign. "
        f"Cap the slowness range at {p_safe:.3g} s/m, low-pass the gather "
        f"below {f_safe:.3g} Hz, or sample the array at "
        f"dx = {dx_safe:.3g} m or finer. Zero-padding cannot help: the "
        f"wavenumber was undersampled before the transform ran.",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def _fk_nfft(nfft, nt, nx):
    """Resolve the zero-padded f-k transform shape ``(NT, NX) >= (nt, nx)``."""
    if nfft is None:
        return nt, nx
    if np.isscalar(nfft):
        NT = NX = int(nfft)
    else:
        nfft = tuple(nfft)
        if len(nfft) != 2:
            raise ConfigurationError(
                "fk_transform: nfft must be an int or (n_time, n_space)"
                f"; got {len(nfft)} entries.")
        NT, NX = int(nfft[0]), int(nfft[1])
    if NT < nt or NX < nx:
        raise ConfigurationError(
            f"fk_transform: nfft {(NT, NX)} must be >= data shape {(nt, nx)} "
            "(zero-pad only, no truncation)")
    return NT, NX


def _require_scalar_geometry(who, sample_rate, dx, signature):
    """Reject an array where a scalar ``sample_rate``/``dx`` belongs.

    ``inverse_taup`` and ``inverse_radon`` take a panel, the scalar geometry
    and then a parameter axis; a call that puts the axis array in a scalar
    slot is caught here and told the expected signature instead of failing
    later inside numpy.
    """
    for name, value in (("sample_rate", sample_rate), ("dx", dx)):
        if np.ndim(value) != 0:
            raise ConfigurationError(
                f"{who}: {name} must be a scalar, got an array of shape "
                f"{np.shape(value)} — check the argument order, which is "
                f"{signature}.")


def _require_axis_array(who, name, value, signature):
    """Reject a parameter axis that is not a 1-D array.

    The axis is the one inverse argument a scalar cannot stand in for: with a
    one-row panel, a scalar in the axis slot lines up in size, so a call
    written in another argument order (a scalar geometry value landing on the
    axis) would run on the wrong numbers. A single-row axis is passed as a
    length-1 array.
    """
    if np.ndim(value) != 1:
        raise ConfigurationError(
            f"{who}: {name} must be a 1-D array (one value per panel "
            f"row; a single row takes a length-1 array); got "
            f"{'a scalar' if np.ndim(value) == 0 else f'shape {np.shape(value)}'}"
            f" — check the argument order, which is {signature}.")


def _flip_wavenumber_axis(F):
    """Negate the spatial-frequency convention of an unshifted 2-D DFT.

    numpy's ``fft2`` carries ``exp(-i2π(ft + νx))``, which puts a wave
    travelling towards +x on ``ω = -c·k``. Reversing the (unshifted) spatial
    axis re-indexes column ``ν`` to hold ``-ν``, so the wave lands on
    ``ω = +c·k`` — the package-wide ``k = ω/c`` convention that
    :func:`taup_transform` and :func:`radon_transform` already use. The map is
    its own inverse, so :func:`inverse_fk` applies the same reversal.
    """
    return np.roll(F[:, ::-1], 1, axis=1)


def _moveout_times(kind, taus, x, m):
    """Moveout time ``t(tau, x; m)`` for the requested Radon kind."""
    if kind == "linear":
        return taus + m * x
    if kind == "parabolic":
        return taus + m * x ** 2
    if kind == "hyperbolic":
        return np.sqrt(taus ** 2 + (x / m) ** 2)
    raise ConfigurationError(
        f"radon: kind must be one of {_RADON_KINDS}, got {kind!r}."
    )


#: Fewest time samples a gather may have with more traces than samples before
#: it reads as ``(nx, nt)``: the channel-count bound ``welch`` applies to a
#: ``(n_samples, n_channels)`` record.
_MAX_TRANSPOSED_NT = 64


def _warn_if_gather_reads_transposed(nt, nx, who):
    """Warn when an ``(nt, nx)`` gather looks like ``(nx, nt)``: a few dozen
    time samples against more traces is a (space, time) array, and every
    transform here would read its traces as time and its samples as space."""
    if nt < nx and nt <= _MAX_TRANSPOSED_NT:
        warnings.warn(
            f"{who}: data is shaped ({nt}, {nx}), read as {nt} time samples "
            f"of {nx} traces, which looks like a (traces, time) array. Pass "
            f"data.T if the first axis is space.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


def radon_transform(data, sample_rate, dx, moveout, kind="linear", x0=0.0):
    """Forward Radon transform (slant stack) of a ``(nt, nx)`` gather.

    Sums the data along moveout curves ``t = t(tau, x)``:

    * ``linear``      ``t = tau + p*x``        (``moveout`` = slowness p, s/m)
    * ``parabolic``   ``t = tau + q*x**2``     (``moveout`` = curvature q, s/m^2)
    * ``hyperbolic``  ``t = sqrt(tau^2+(x/v)^2)`` (``moveout`` = velocity v, m/s)

    Parameters
    ----------
    data : ndarray
        Gather, shape ``(nt, nx)`` (time down columns, offset across rows).
    sample_rate : float
        Temporal sample rate (Hz).
    dx : float
        Sensor spacing (m).
    moveout : array
        Moveout parameters to scan (units per ``kind`` above).
    kind : {'linear', 'parabolic', 'hyperbolic'}
        Moveout family. ``'linear'`` is the tau-p slant stack.
    x0 : float
        Reference offset (m) subtracted from sensor positions.

    Returns
    -------
    RadonResult
        Namedtuple ``(moveout, taus, panel)``: the scanned moveout axis, the
        intercept-time axis (s), and the Radon panel ``(len(moveout), nt)``.
    """
    d = require_real_signal(
        data, "radon_transform", ndim=2, shape_hint=" (nt, nx)",
        why="; a complex gather's imaginary part is what a real-valued "
            "stack would discard.",
        remediation="Stack the real and imaginary parts separately "
                    "(the transform is linear), or use fk_transform, "
                    "which accepts complex data.")
    nt, nx = d.shape
    _warn_if_gather_reads_transposed(nt, nx, "radon_transform")
    sample_rate = require_positive_finite_scalar(
        sample_rate, "radon_transform", "sample_rate", " Hz")
    dx = require_positive_finite_scalar(dx, "radon_transform", "dx", " m")
    x = np.arange(nx) * dx - float(x0)
    moveout = np.atleast_1d(np.asarray(moveout, dtype=float))
    if kind == "hyperbolic" and np.any(moveout <= 0):
        raise ConfigurationError(
            "radon_transform: hyperbolic moveout is a velocity (m/s) and the "
            "moveout curve sqrt(tau^2 + (x/v)^2) divides by it, so every "
            f"value must be > 0; got min {moveout.min()}.")
    taus = np.arange(nt) / float(sample_rate)
    R = np.zeros((moveout.size, nt))
    for i, m in enumerate(moveout):
        for ix in range(nx):
            tt = _moveout_times(kind, taus, x[ix], m)
            R[i] += np.interp(tt, taus, d[:, ix], left=0.0, right=0.0)
    return RadonResult(moveout, taus, R, kind=kind)


def inverse_radon(R, sample_rate, dx, moveout, nx, kind="linear", x0=0.0):
    """Adjoint (back-projection) Radon transform: ``(n_moveout, nt) -> (nt, nx)``.

    Spreads each Radon sample back along its moveout curve. This is the matched
    adjoint, not a least-squares inverse, so a forward-then-adjoint round trip is
    band-limited, not exact. It is the exact transpose of
    :func:`radon_transform` for every ``kind`` — ``<L x, y> == <x, A y>`` to
    machine precision — which is what an iterative least-squares (sparse Radon)
    solver needs.

    ``sample_rate``/``dx`` are the geometry and ``moveout`` the scanned
    parameter axis — the order :func:`inverse_taup` shares.

    Parameters
    ----------
    R : ndarray
        The ``(n_moveout, nt)`` Radon panel.
    sample_rate : float
        Temporal sample rate (Hz).
    dx : float
        Sensor spacing (m).
    moveout : array_like
        The scanned parameter axis.
    nx : int
        Traces in the output gather.
    kind : {'linear', 'parabolic', 'hyperbolic'}, optional
        The moveout family. Default ``'linear'``.
    x0 : float, optional
        Reference offset (m) of the forward transform. Default 0.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ConfigurationError(
            "inverse_radon: R must be 2-D (n_moveout, nt)"
            f"; got shape {R.shape}.")
    nm, nt = R.shape
    _require_scalar_geometry(
        "inverse_radon", sample_rate, dx,
        "inverse_radon(R, sample_rate, dx, moveout, nx)")
    _require_axis_array("inverse_radon", "moveout", moveout,
                        "inverse_radon(R, sample_rate, dx, moveout, nx)")
    moveout = np.asarray(moveout, dtype=float)
    if moveout.size != nm:
        raise ConfigurationError(
            f"inverse_radon: moveout length ({moveout.size}) must match R rows "
            f"({nm}); the signature is inverse_radon(R, sample_rate, dx, "
            "moveout, nx) — the moveout axis comes fourth, as the slowness "
            "axis does in inverse_taup.")
    if kind == "hyperbolic" and np.any(moveout <= 0):
        raise ConfigurationError(
            "inverse_radon: hyperbolic moveout is a velocity (m/s) and the "
            "moveout curve sqrt(tau^2 + (x/v)^2) divides by it, so every "
            f"value must be > 0; got min {moveout.min()}.")
    fs = float(sample_rate)
    taus = np.arange(nt) / fs
    x = np.arange(int(nx)) * float(dx) - float(x0)
    out = np.zeros((nt, int(nx)))
    for i, m in enumerate(moveout):
        for ix in range(int(nx)):
            # Scatter: each Radon sample is split between the two grid samples
            # straddling t(tau, x) with the same weights the forward
            # `np.interp` gives them, which is the transpose of that
            # interpolation. Gathering instead — reading the curve back with
            # `np.interp(taus, tt, ...)` — only coincides with the transpose
            # when the moveout is a pure time shift (linear, parabolic); a
            # hyperbolic curve compresses near tau=0 and the gather returns
            # early samples at a fraction of their forward weight.
            idx = _moveout_times(kind, taus, x[ix], m) * fs
            inside = (idx >= 0.0) & (idx <= nt - 1)
            j = np.floor(idx[inside]).astype(int)
            w = idx[inside] - j
            r = R[i][inside]
            out[:, ix] += np.bincount(j, weights=(1.0 - w) * r,
                                      minlength=nt)[:nt]
            out[:, ix] += np.bincount(j + 1, weights=w * r,
                                      minlength=nt + 1)[:nt]
    return out


def taup_transform(data, sample_rate, dx, slownesses=None, n_slowness=201,
                   p_max=None, *, x0=0.0, window=None, nfft=None):
    """Forward linear tau-p (slant stack), frequency-domain.

    Returns a :class:`TauPResult` namedtuple ``(slownesses, taus, panel)``:
    slowness axis (s/m), intercept-time axis (s), and the panel ``(n_slowness,
    NT)``.

    ``x0`` is the reference offset (m) subtracted from the sensor positions, as
    in :func:`radon_transform` — it walks the same moveout curve
    ``t = tau + p*(x - x0)``.

    Built from FFT products, the transform is **circular in tau**: each trace
    is read at ``(tau + p*(x - x0)) mod (NT/fs)``, ``NT`` being the time-FFT
    length. An intercept ``t0 - p*(x - x0)`` outside the ``[0, NT/fs)`` window
    (either sign) therefore stacks at full amplitude at that intercept
    **modulo** ``NT/fs`` — a tau where :func:`radon_transform`
    (``kind='linear'``, same ``x0``), which drops out-of-window samples, is
    exactly zero.

    An in-window intercept is necessary for the two to agree but not
    sufficient: the shift here is a spectral phase ramp (band-limited) while
    the Radon panel interpolates each trace linearly, so a moveout of a
    non-integer number of samples per trace separates them even in-window —
    measured 9.1% of the peak at ``p*dx*fs = 2.5``. Expect exact agreement
    only for a whole-sample moveout.

    ``window`` is a temporal :func:`scipy.signal.get_window` spec (name or
    ``(name, *params)`` tuple) applied down each trace before the time FFT to
    curb leakage; ``None`` is rectangular. ``nfft`` zero-pads the time FFT to
    ``NT >= nt`` samples; ``None`` keeps ``nt``. The ``tau`` spacing is
    ``1/fs`` either way — zero-padding does not refine it, it **lengthens**
    the tau axis into a guard band: an intercept up to ``(NT - nt)/fs`` s
    outside the record lands in the padded ``[nt/fs, NT/fs)`` rows instead of
    aliasing in among the physical taus.

    Parameters
    ----------
    data : ndarray
        Gather ``(nt, nx)``.
    sample_rate : float
        Temporal sample rate (Hz).
    dx : float
        Sensor spacing (m).
    slownesses : array_like, optional
        The slowness axis (s/m); ``None`` builds one from ``n_slowness`` and
        ``p_max``.
    n_slowness : int, optional
        Slownesses of the built axis. Default 201.
    p_max : float, optional
        Largest slowness (s/m) of the built axis.
    x0 : float, optional
        Reference offset (m). Default 0.
    window : str or tuple, optional
        Temporal taper (see below); ``None`` is rectangular.
    nfft : int, optional
        Time-FFT length (see below); ``None`` keeps ``nt``.

    The spatial axis has no such knob, and deliberately so
    -----------------------------------------------------
    The slowness axis is yours to set outright — ``slownesses`` takes any
    array, uniform or not, and ``n_slowness``/``p_max`` build one for you — so
    resolution in ``p`` is never limited by a transform length the way
    :func:`fk_transform`'s wavenumber axis is. There is correspondingly no
    spatial ``nfft``: this transform SUMS over sensors rather than
    transforming across them, so appending zero traces adds zero terms to that
    sum and changes nothing, bit for bit. (In ``fk_transform`` the spatial FFT
    length sets the wavenumber grid ``dk = 2*pi/(NX*dx)``, which is why the
    knob is real there and meaningless here.)

    What the array spacing DOES limit is aliasing. Between adjacent traces the
    stack sees ``2*pi*f*p*dx`` modulo a turn, so past ``|p| = 1/(2*f*dx)`` it
    cannot separate ``p`` from ``p -/+ 1/(f*dx)``: a 900 Hz plane wave at
    ``p = +4e-4`` s/m on a ``dx = 2`` m array surfaces at ``-1.55e-4`` — the
    wrong slowness AND the wrong direction of travel. A warning fires when the
    requested range crosses that bound, judged against the frequency holding
    99% of the record's energy rather than the Nyquist rate so a narrowband
    gather is measured on the band it actually occupies. Heed it upstream: no
    padding recovers a wavenumber the array never sampled.
    """
    d = require_real_signal(
        data, "taup_transform", ndim=2, shape_hint=" (nt, nx)",
        why="; a complex gather's imaginary part is what a real-valued "
            "stack would discard.",
        remediation="Stack the real and imaginary parts separately "
                    "(the transform is linear), or use fk_transform, "
                    "which accepts complex data.")
    nt, nx = d.shape
    _warn_if_gather_reads_transposed(nt, nx, "taup_transform")
    fs = require_positive_finite_scalar(sample_rate, "taup_transform",
                                        "sample_rate", " Hz")
    dx = require_positive_finite_scalar(dx, "taup_transform", "dx", " m")
    NT = nt if nfft is None else int(nfft)
    if NT < nt:
        raise ConfigurationError(
            f"taup_transform: nfft ({NT}) must be >= nt ({nt}) (zero-pad only)")
    d = d * _spectral_window(window, nt)[:, None]
    x = np.arange(nx) * dx - float(x0)
    if slownesses is None:
        if p_max is None:
            # +/- 1e-3 s/m: everything with an apparent velocity above
            # 1000 m/s, which spans the water column and most sediments.
            p_max = 1.0 / 1000.0
        slownesses = np.linspace(-p_max, p_max, int(n_slowness))
    slownesses = np.atleast_1d(np.asarray(slownesses, dtype=float))
    D = np.fft.rfft(d, n=NT, axis=0)
    freqs = np.fft.rfftfreq(NT, 1.0 / fs)                # Hz
    omega = 2.0 * np.pi * freqs                          # rad/s
    _warn_spatial_aliasing(D, freqs, dx, slownesses, "taup_transform")
    taup = np.empty((slownesses.size, NT))
    for i, p in enumerate(slownesses):
        # Sign: numpy's forward transform carries exp(-j*omega*t), so a
        # +exp(j*omega*p*x) factor advances trace x by p*x. Summing over x is
        # then u(tau, p) = sum_x d(tau + p*x, x) — the same moveout curve
        # `radon_transform(kind='linear')` interpolates in the time domain.
        phase = np.exp(1j * omega[:, None] * (p * x)[None, :])
        taup[i] = np.fft.irfft(np.sum(D * phase, axis=1), n=NT)
    return TauPResult(slownesses, np.arange(NT) / fs, taup)


def inverse_taup(taup, sample_rate, dx, slownesses, nx, *, x0=0.0):
    """Adjoint slant stack ``(n_slowness, nt) -> (nt, nx)``.

    Standalone inverse — pass a tau-p panel you already have (e.g. a filtered
    one) plus its slowness axis and geometry; no prior :func:`taup_transform`
    call needed. ``x0`` is the reference offset (m) the forward transform used;
    pass the same value back.

    The argument order is :func:`inverse_radon`'s and the forward
    transforms': panel, ``sample_rate``, ``dx``, then the parameter axis
    (``slownesses``) and ``nx``. The scalar geometry arguments are
    type-checked, so a call that puts the slowness array in a scalar slot is
    refused by name.

    Parameters
    ----------
    taup : ndarray
        The ``(n_slowness, nt)`` tau-p panel.
    sample_rate : float
        Temporal sample rate (Hz).
    dx : float
        Sensor spacing (m).
    slownesses : array_like
        The panel's slowness axis (s/m).
    nx : int
        Traces in the output gather.
    x0 : float, optional
        Reference offset (m) of the forward transform. Default 0.
    """
    u = np.asarray(taup, dtype=float)
    if u.ndim != 2:
        raise ConfigurationError(
            "inverse_taup: taup must be 2-D (n_slowness, nt)"
            f"; got shape {u.shape}.")
    n_p, nt = u.shape
    _require_scalar_geometry("inverse_taup", sample_rate, dx,
                             "inverse_taup(taup, sample_rate, dx, slownesses, nx)")
    _require_axis_array("inverse_taup", "slownesses", slownesses,
                        "inverse_taup(taup, sample_rate, dx, slownesses, nx)")
    slownesses = np.asarray(slownesses, dtype=float)
    if slownesses.size != n_p:
        raise ConfigurationError(
            f"inverse_taup: slownesses length ({slownesses.size}) must match "
            f"taup rows ({n_p}); the signature is inverse_taup(taup, "
            "sample_rate, dx, slownesses, nx) — the slowness axis comes "
            "fourth, as the moveout axis does in inverse_radon.")
    x = np.arange(int(nx)) * float(dx) - float(x0)
    U = np.fft.rfft(u, axis=1)
    omega = 2.0 * np.pi * np.fft.rfftfreq(nt, 1.0 / float(sample_rate))
    D = np.zeros((omega.size, int(nx)), dtype=complex)
    for i, p in enumerate(slownesses):
        # Conjugate phase of the forward transform (the adjoint): each slowness
        # is spread back along its own moveout, delayed by p*x.
        D += U[i][:, None] * np.exp(-1j * omega[:, None] * (p * x)[None, :])
    return np.fft.irfft(D, n=nt, axis=0)


def inverse_fk(FK):
    """Inverse f-k transform: complex (fftshifted) spectrum -> real gather.

    Pass the (possibly filtered/muted) complex spectrum — i.e. the ``spectrum``
    returned by :func:`fk_transform` (single-segment), after any f-k mask. It
    must be in the ``fftshift``ed layout that :func:`fk_transform` produces.

    The output has the **spectrum's** shape ``(NT, NX)`` — the zero-padded
    ``nfft`` shape when the forward transform was padded, with the original
    ``(nt, nx)`` gather in its top-left corner followed by the padding.
    The forward ``window`` taper is **not** undone: a windowed forward
    transform inverts to the *tapered* gather, and recovering the original
    data requires dividing the tapers back out (undefined where they are
    zero). For an exact round trip run ``fk_transform`` with ``window=None``.

    Parameters
    ----------
    FK : ndarray
        The complex, fftshifted spectrum of :func:`fk_transform`.
    """
    if FK is None:
        raise ConfigurationError(
            "inverse_fk: spectrum is None — an f-k panel averaged over more "
            "than one segment has no phase and cannot be inverted. Re-run "
            "fk_transform with nperseg=None for an invertible spectrum.")
    if isinstance(FK, tuple):
        raise ConfigurationError(
            "inverse_fk: pass the complex spectrum (the .spectrum field / 4th "
            "element of fk_transform's result), not the whole FKResult tuple.")
    fk = np.asarray(FK)
    if fk.ndim != 2:
        raise ConfigurationError(
            "inverse_fk: FK must be 2-D (nt, nx)"
            f"; got shape {fk.shape}.")
    return np.real(np.fft.ifft2(
        _flip_wavenumber_axis(np.fft.ifftshift(fk, axes=(0, 1)))))


def fk_transform(data, sample_rate, dx, *, nperseg=None, noverlap=None,
                 window=None, nfft=None, scaling='power'):
    """Frequency-wavenumber transform with optional Welch time-averaging.

    Returns an :class:`FKResult` namedtuple ``(frequencies, wavenumbers, power,
    spectrum)``. ``frequencies`` are in Hz; ``wavenumbers`` is the **angular**
    wavenumber ``k = 2π·ν`` in **rad/m** (the package-wide ``k = ω/c``
    convention), so a wave travelling towards +x at speed ``c`` sits on the
    line ``ω = +c·k`` — the same sign as the apparent slowness ``p = +1/c``
    that :func:`taup_transform` and :func:`radon_transform` report for it.
    The spatial axis is therefore the negative of raw ``np.fft.fft2``
    indexing, whose ``exp(-i2πνx)`` kernel would place that wave on
    ``ω = -c·k``; directional f-k muting must use this sign.
    ``power`` is the real ``|FK|^2`` panel (fftshifted); when ``scaling='density'``
    it is a PSD density per ``Hz·rad/m`` with ``ΣP·Δf·Δk = ⟨x²⟩``, and the
    result's ``scaling`` attribute reads ``'density'``; with ``scaling='power'``
    (the default) it is the raw squared magnitude of the windowed, zero-padded
    FFT, which grows with the record size and carries no physical unit, and
    ``scaling`` reads ``'power'``. :func:`~uacpy.plot.plot_fk` labels
    the panel from that attribute. Whenever the
    settings yield a single segment (``nperseg=None``, i.e. the whole record, or
    an ``nperseg``/``noverlap`` pair that fits only one block) ``spectrum`` is
    that segment's complex (fftshifted) panel for :func:`inverse_fk`. With
    several segments the time axis is split into overlapping blocks, ``|FK|^2``
    is averaged across them (variance ~1/N, standard deviation ~1/sqrt(N); a
    single-snapshot f-k panel is
    an inconsistent estimator), and ``spectrum`` is ``None`` — an averaged power
    panel has no single phase and is not invertible.

    Parameters
    ----------
    data : ndarray
        Gather ``(nt, nx)``.
    sample_rate : float
        Temporal sample rate (Hz).
    dx : float
        Sensor spacing (m).
    nperseg : int, optional
        Time-segment length for Welch averaging. ``None`` (default) uses the
        whole record (one segment, invertible).
    noverlap : int, optional
        Overlap between segments (samples). Defaults to ``nperseg // 2`` when
        ``nperseg`` is set; ``0`` otherwise. Must satisfy ``0 <= noverlap < nperseg``.
    window, nfft, scaling
        As in the single-segment transform, applied per segment.
    """
    if scaling not in FK_SCALINGS:
        raise ConfigurationError(
            f"fk_transform: scaling must be one of {FK_SCALINGS}; got "
            f"{scaling!r}.")
    d = np.asarray(data)
    if d.ndim != 2:
        raise ConfigurationError(
            "fk_transform: data must be 2-D (nt, nx)"
            f"; got shape {d.shape}.")
    nt, nx = d.shape
    _warn_if_gather_reads_transposed(nt, nx, "fk_transform")
    fs = require_positive_finite_scalar(
        sample_rate, "fk_transform", "sample_rate", " Hz")
    dx = require_positive_finite_scalar(dx, "fk_transform", "dx", " m")
    seg = nt if nperseg is None else int(nperseg)
    if seg > nt or seg < 1:
        raise ConfigurationError(
            f"fk_transform: nperseg ({seg}) must be in [1, nt={nt}].")
    ov = (_default_noverlap(window, seg)
          if (noverlap is None and nperseg is not None) else int(noverlap or 0))
    if not (0 <= ov < seg):
        raise ConfigurationError(
            f"fk_transform: noverlap ({ov}) must be in [0, nperseg={seg})")
    wt, wx = _fk_tapers(window, seg, nx)
    NF, NX = _fk_nfft(nfft, seg, nx)

    # Whole-segment starts only (trailing samples that don't fill a segment are
    # dropped, as in scipy's Welch); each block is therefore exactly `seg` long.
    starts = range(0, nt - seg + 1, seg - ov)
    power = np.zeros((NF, NX))
    last_spectrum = None
    n_seg = 0
    for s0 in starts:
        block = d[s0:s0 + seg]
        bw = block * wt[:, None] * wx[None, :]
        FKc = np.fft.fftshift(
            _flip_wavenumber_axis(np.fft.fft2(bw, s=(NF, NX))), axes=(0, 1))
        last_spectrum = FKc
        FKp = np.abs(FKc) ** 2
        if scaling == 'density':
            s2 = float(np.sum(wt ** 2) * np.sum(wx ** 2))
            # Density per (Hz · rad/m): the extra 2π converts the per-bin spatial
            # width to rad/m so that ΣP·Δf·Δk = ⟨x²⟩ still holds with k in rad/m.
            FKp = FKp * (float(dx) / (fs * s2 * 2.0 * np.pi))
        power += FKp
        n_seg += 1
    power /= n_seg

    freqs = np.fft.fftshift(np.fft.fftfreq(NF, d=1.0 / fs))
    # Angular wavenumber k = 2π·ν in rad/m (ν = fftfreq is cycles/m), matching
    # the package-wide convention k = ω/c used by the models: a wave of speed c
    # lies on the line ω = c·k (i.e. f = c·k/2π — the acoustic "sound cone").
    # The axis needs no negation here because `_flip_wavenumber_axis` already
    # re-indexed the panel columns onto it.
    wavenumbers = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, d=dx))
    spectrum = last_spectrum if n_seg == 1 else None
    return FKResult(freqs, wavenumbers, power, spectrum, scaling=scaling)
