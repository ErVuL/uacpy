"""Wavenumber/slowness transforms of (t-x) gathers: f-k, tau-p, Radon.

Duals that decompose a time-offset gather by apparent slowness / wavenumber.
Each has a standalone reverse map, so the workflow is forward -> filter the
coefficients -> reverse. Only :func:`inverse_fk` is a true inverse (an
``ifft2``); :func:`inverse_taup` and :func:`inverse_radon` are the matched
*adjoints* (back-projection), so a forward-then-reverse round trip is
band-limited, not exact.

All three agree on sign: a wave travelling towards +x at speed ``c`` has
apparent slowness ``p = +1/c`` in the tau-p and Radon panels and sits at
``k = +ω/c`` in the f-k panel. The f-k spatial axis is therefore the negative
of raw ``np.fft.fft2`` indexing (see :func:`_flip_wavenumber_axis`).
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy.signal import get_window

from uacpy.core.exceptions import ConfigurationError
from uacpy.acoustic_signal._signal_validate import require_positive_finite_scalar


_RADON_KINDS = ("linear", "parabolic", "hyperbolic")

RadonResult = namedtuple("RadonResult", "moveout taus panel")
TauPResult = namedtuple("TauPResult", "slownesses taus panel")
FKResult = namedtuple("FKResult", "frequencies wavenumbers power spectrum")


def _taper(spec, n):
    """Length-``n`` taper for a ``scipy.signal.get_window`` spec.

    ``spec`` is ``None`` (rectangular ``ones``) or any get_window argument —
    a name (``'hann'``) or a ``(name, *params)`` tuple (``('kaiser', 8)``).
    Periodic (``fftbins=True``) form, the correct convention for spectra.
    """
    if spec is None:
        return np.ones(int(n))
    return get_window(spec, int(n), fftbins=True).astype(float)


def _fk_tapers(window, nt, nx):
    """Separable (time, space) tapers for the 2-D f-k window.

    ``window`` applies one spec to both axes; a 2-element ``list``
    ``[time_spec, space_spec]`` tapers the axes independently.
    """
    if isinstance(window, list):
        if len(window) != 2:
            raise ConfigurationError(
                "fk_transform: window list must be [time_window, space_window]"
                f"; got {len(window)} entries")
        t_spec, x_spec = window
    else:
        t_spec = x_spec = window
    return _taper(t_spec, nt), _taper(x_spec, nx)


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
                f"; got {len(nfft)} entries")
        NT, NX = int(nfft[0]), int(nfft[1])
    if NT < nt or NX < nx:
        raise ConfigurationError(
            f"fk_transform: nfft {(NT, NX)} must be >= data shape {(nt, nx)} "
            "(zero-pad only, no truncation)")
    return NT, NX


def _require_scalar_geometry(caller, sample_rate, dx, signature):
    """Reject an array where a scalar ``sample_rate``/``dx`` belongs.

    ``inverse_taup`` and ``inverse_radon`` both take a panel, a parameter axis
    and the scalar geometry, but in different positions, so passing one's
    argument order to the other lands an array on a scalar slot. Catching it
    here names the expected signature instead of failing later inside numpy.
    """
    for name, value in (("sample_rate", sample_rate), ("dx", dx)):
        if np.ndim(value) != 0:
            raise ConfigurationError(
                f"{caller}: {name} must be a scalar, got an array of shape "
                f"{np.shape(value)} — check the argument order, which is "
                f"{signature}.")


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
        f"radon: kind must be one of {_RADON_KINDS}, got {kind!r}"
    )


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
    d = np.asarray(data, dtype=float)
    if d.ndim != 2:
        raise ConfigurationError(
            "radon_transform: data must be 2-D (nt, nx)"
            f"; got shape {d.shape}")
    nt, nx = d.shape
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
    return RadonResult(moveout, taus, R)


def inverse_radon(R, sample_rate, dx, moveout, nx, kind="linear", x0=0.0):
    """Adjoint (back-projection) Radon transform: ``(n_moveout, nt) -> (nt, nx)``.

    Spreads each Radon sample back along its moveout curve. This is the matched
    adjoint, not a least-squares inverse, so a forward-then-adjoint round trip is
    band-limited, not exact. It is the exact transpose of
    :func:`radon_transform` for every ``kind`` — ``<L x, y> == <x, A y>`` to
    machine precision — which is what an iterative least-squares (sparse Radon)
    solver needs.

    ``sample_rate``/``dx`` are the geometry and ``moveout`` the scanned
    parameter axis; note the order differs from :func:`inverse_taup`, whose
    slowness axis comes second.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ConfigurationError(
            "inverse_radon: R must be 2-D (n_moveout, nt)"
            f"; got shape {R.shape}")
    nm, nt = R.shape
    _require_scalar_geometry(
        "inverse_radon", sample_rate, dx,
        "inverse_radon(R, sample_rate, dx, moveout, nx)")
    moveout = np.atleast_1d(np.asarray(moveout, dtype=float))
    if moveout.size != nm:
        raise ConfigurationError(
            f"inverse_radon: moveout length ({moveout.size}) must match R rows "
            f"({nm}); the signature is inverse_radon(R, sample_rate, dx, "
            "moveout, nx) — the moveout axis comes fourth, unlike "
            "inverse_taup where the slowness axis comes second.")
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
    ``t = tau + p*(x - x0)``, so the two agree only when both are given the
    same ``x0``. ``window`` is a temporal :func:`scipy.signal.get_window` spec
    (name or ``(name, *params)`` tuple) applied down each trace before the time
    FFT to curb leakage; ``None`` is rectangular. ``nfft`` zero-pads the time
    FFT to ``NT >= nt`` samples (finer ``tau`` spacing); ``None`` keeps ``nt``.
    """
    d = np.asarray(data, dtype=float)
    if d.ndim != 2:
        raise ConfigurationError(
            "taup_transform: data must be 2-D (nt, nx)"
            f"; got shape {d.shape}")
    nt, nx = d.shape
    fs = require_positive_finite_scalar(sample_rate, "taup_transform",
                                        "sample_rate", " Hz")
    dx = require_positive_finite_scalar(dx, "taup_transform", "dx", " m")
    NT = nt if nfft is None else int(nfft)
    if NT < nt:
        raise ConfigurationError(
            f"taup_transform: nfft ({NT}) must be >= nt ({nt}) (zero-pad only)")
    d = d * _taper(window, nt)[:, None]
    x = np.arange(nx) * dx - float(x0)
    if slownesses is None:
        if p_max is None:
            # +/- 1e-3 s/m: everything with an apparent velocity above
            # 1000 m/s, which spans the water column and most sediments.
            p_max = 1.0 / 1000.0
        slownesses = np.linspace(-p_max, p_max, int(n_slowness))
    slownesses = np.atleast_1d(np.asarray(slownesses, dtype=float))
    D = np.fft.rfft(d, n=NT, axis=0)
    omega = 2.0 * np.pi * np.fft.rfftfreq(NT, 1.0 / fs)  # rad/s
    taup = np.empty((slownesses.size, NT))
    for i, p in enumerate(slownesses):
        # Sign: numpy's forward transform carries exp(-j*omega*t), so a
        # +exp(j*omega*p*x) factor advances trace x by p*x. Summing over x is
        # then u(tau, p) = sum_x d(tau + p*x, x) — the same moveout curve
        # `radon_transform(kind='linear')` interpolates in the time domain.
        phase = np.exp(1j * omega[:, None] * (p * x)[None, :])
        taup[i] = np.fft.irfft(np.sum(D * phase, axis=1), n=NT)
    return TauPResult(slownesses, np.arange(NT) / fs, taup)


def inverse_taup(taup, slownesses, sample_rate, dx, nx, *, x0=0.0):
    """Adjoint slant stack ``(n_slowness, nt) -> (nt, nx)``.

    Standalone inverse — pass a tau-p panel you already have (e.g. a filtered
    one) plus its slowness axis and geometry; no prior :func:`taup_transform`
    call needed. ``x0`` is the reference offset (m) the forward transform used;
    pass the same value back.

    The slowness axis comes **second** here, while :func:`inverse_radon` takes
    its moveout axis **fourth** (after ``sample_rate`` and ``dx``). Both are
    positional arrays, so the two orders are not interchangeable; the scalar
    geometry arguments are type-checked below to catch the swap.
    """
    u = np.asarray(taup, dtype=float)
    if u.ndim != 2:
        raise ConfigurationError(
            "inverse_taup: taup must be 2-D (n_slowness, nt)"
            f"; got shape {u.shape}")
    n_p, nt = u.shape
    _require_scalar_geometry("inverse_taup", sample_rate, dx,
                             "inverse_taup(taup, slownesses, sample_rate, dx, nx)")
    slownesses = np.atleast_1d(np.asarray(slownesses, dtype=float))
    if slownesses.size != n_p:
        raise ConfigurationError(
            f"inverse_taup: slownesses length ({slownesses.size}) must match "
            f"taup rows ({n_p}); the signature is inverse_taup(taup, "
            "slownesses, sample_rate, dx, nx) — the slowness axis comes "
            "second, unlike inverse_radon where it comes fourth.")
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
            f"; got shape {fk.shape}")
    return np.real(np.fft.ifft2(
        _flip_wavenumber_axis(np.fft.ifftshift(fk, axes=(0, 1)))))


def fk_transform(data, sample_rate, dx, *, nperseg=None, noverlap=None,
                 window=None, nfft=None, normalize=False):
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
    ``power`` is the real ``|FK|^2`` panel (fftshifted); when ``normalize=True``
    it is a PSD density per ``Hz·rad/m`` with ``ΣP·Δf·Δk = ⟨x²⟩``. Whenever the
    settings yield a single segment (``nperseg=None``, i.e. the whole record, or
    an ``nperseg``/``noverlap`` pair that fits only one block) ``spectrum`` is
    that segment's complex (fftshifted) panel for :func:`inverse_fk`. With
    several segments the time axis is split into overlapping blocks, ``|FK|^2``
    is averaged across them (variance ~1/sqrt(N); a single-snapshot f-k panel is
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
    window, nfft, normalize
        As in the single-segment transform, applied per segment.
    """
    d = np.asarray(data)
    if d.ndim != 2:
        raise ConfigurationError(
            "fk_transform: data must be 2-D (nt, nx)"
            f"; got shape {d.shape}")
    nt, nx = d.shape
    fs = require_positive_finite_scalar(
        sample_rate, "fk_transform", "sample_rate", " Hz")
    dx = require_positive_finite_scalar(dx, "fk_transform", "dx", " m")
    seg = nt if nperseg is None else int(nperseg)
    if seg > nt or seg < 1:
        raise ConfigurationError(
            f"fk_transform: nperseg ({seg}) must be in [1, nt={nt}]")
    ov = (seg // 2) if (noverlap is None and nperseg is not None) else int(noverlap or 0)
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
        if normalize:
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
    return FKResult(freqs, wavenumbers, power, spectrum)
