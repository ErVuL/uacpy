"""Wavenumber/slowness transforms of (t-x) gathers: f-k, tau-p, Radon.

Duals that decompose a time-offset gather by apparent slowness / wavenumber.
Each has a standalone reverse map, so the workflow is forward -> filter the
coefficients -> reverse. Only :func:`inverse_fk` is a true inverse (an
``ifft2``); :func:`inverse_taup` and :func:`inverse_radon` are the matched
*adjoints* (back-projection), so a forward-then-reverse round trip is
band-limited, not exact.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy.signal import get_window

from uacpy.core.exceptions import ConfigurationError


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
                "fk_transform: window list must be [time_window, space_window]")
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
                "fk_transform: nfft must be an int or (n_time, n_space)")
        NT, NX = int(nfft[0]), int(nfft[1])
    if NT < nt or NX < nx:
        raise ConfigurationError(
            f"fk_transform: nfft {(NT, NX)} must be >= data shape {(nt, nx)} "
            "(zero-pad only, no truncation)")
    return NT, NX


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
        raise ConfigurationError("radon_transform: data must be 2-D (nt, nx)")
    nt, nx = d.shape
    x = np.arange(nx) * float(dx) - float(x0)
    moveout = np.atleast_1d(np.asarray(moveout, dtype=float))
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
    band-limited, not exact.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ConfigurationError("inverse_radon: R must be 2-D (n_moveout, nt)")
    nm, nt = R.shape
    moveout = np.atleast_1d(np.asarray(moveout, dtype=float))
    if moveout.size != nm:
        raise ConfigurationError("inverse_radon: moveout length must match R rows")
    taus = np.arange(nt) / float(sample_rate)
    x = np.arange(int(nx)) * float(dx) - float(x0)
    out = np.zeros((nt, int(nx)))
    for i, m in enumerate(moveout):
        for ix in range(int(nx)):
            tt = _moveout_times(kind, taus, x[ix], m)
            out[:, ix] += np.interp(taus, tt, R[i], left=0.0, right=0.0)
    return out


def taup_transform(data, sample_rate, dx, slownesses=None, n_slowness=201,
                   p_max=None, *, window=None, nfft=None):
    """Forward linear tau-p (slant stack), frequency-domain.

    Returns a :class:`TauPResult` namedtuple ``(slownesses, taus, panel)``:
    slowness axis (s/m), intercept-time axis (s), and the panel ``(n_slowness,
    NT)``.

    ``window`` is a temporal :func:`scipy.signal.get_window` spec (name or
    ``(name, *params)`` tuple) applied down each trace before the time FFT to
    curb leakage; ``None`` is rectangular. ``nfft`` zero-pads the time FFT to
    ``NT >= nt`` samples (finer ``tau`` spacing); ``None`` keeps ``nt``.
    """
    d = np.asarray(data, dtype=float)
    if d.ndim != 2:
        raise ConfigurationError("taup_transform: data must be 2-D (nt, nx)")
    nt, nx = d.shape
    fs = float(sample_rate)
    NT = nt if nfft is None else int(nfft)
    if NT < nt:
        raise ConfigurationError(
            f"taup_transform: nfft ({NT}) must be >= nt ({nt}) (zero-pad only)")
    d = d * _taper(window, nt)[:, None]
    x = np.arange(nx) * float(dx)
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


def inverse_taup(taup, slownesses, sample_rate, dx, nx):
    """Adjoint slant stack ``(n_slowness, nt) -> (nt, nx)``.

    Standalone inverse — pass a tau-p panel you already have (e.g. a filtered
    one) plus its slowness axis and geometry; no prior :func:`taup_transform`
    call needed.
    """
    u = np.asarray(taup, dtype=float)
    if u.ndim != 2:
        raise ConfigurationError("inverse_taup: taup must be 2-D (n_slowness, nt)")
    n_p, nt = u.shape
    slownesses = np.atleast_1d(np.asarray(slownesses, dtype=float))
    if slownesses.size != n_p:
        raise ConfigurationError("inverse_taup: slownesses length must match taup rows")
    x = np.arange(int(nx)) * float(dx)
    U = np.fft.rfft(u, axis=1)
    omega = 2.0 * np.pi * np.fft.rfftfreq(nt, 1.0 / float(sample_rate))
    D = np.zeros((omega.size, int(nx)), dtype=complex)
    for i, p in enumerate(slownesses):
        # Conjugate phase of the forward transform (the adjoint): each slowness
        # is spread back along its own moveout, delayed by p*x.
        D += U[i][:, None] * np.exp(-1j * omega[:, None] * (p * x)[None, :])
    return np.fft.irfft(D, n=nt, axis=0)


def inverse_fk(FK):
    """Inverse f-k transform: complex (fftshifted) spectrum -> real ``(nt, nx)``.

    Pass the (possibly filtered/muted) complex spectrum — i.e. the ``spectrum``
    returned by :func:`fk_transform` (single-segment), after any f-k mask. It
    must be in the ``fftshift``ed layout that :func:`fk_transform` produces.
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
        raise ConfigurationError("inverse_fk: FK must be 2-D (nt, nx)")
    return np.real(np.fft.ifft2(np.fft.ifftshift(fk, axes=(0, 1))))


def fk_transform(data, sample_rate, dx, *, nperseg=None, noverlap=None,
                 window=None, nfft=None, normalize=False):
    """Frequency-wavenumber transform with optional Welch time-averaging.

    Returns an :class:`FKResult` namedtuple ``(frequencies, wavenumbers, power,
    spectrum)``. ``frequencies`` are in Hz; ``wavenumbers`` is the **angular**
    wavenumber ``k = 2π·ν`` in **rad/m** (the package-wide ``k = ω/c``
    convention), so a wave of speed ``c`` sits on the line ``ω = c·k``.
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
        raise ConfigurationError("fk_transform: data must be 2-D (nt, nx)")
    nt, nx = d.shape
    fs = float(sample_rate)
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
        FKc = np.fft.fftshift(np.fft.fft2(bw, s=(NF, NX)), axes=(0, 1))
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
    wavenumbers = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, d=dx))
    spectrum = last_spectrum if n_seg == 1 else None
    return FKResult(freqs, wavenumbers, power, spectrum)
