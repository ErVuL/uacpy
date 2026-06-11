"""Wavenumber/slowness transforms of (t-x) gathers: f-k, tau-p, Radon.

Duals that decompose a time-offset gather by apparent slowness / wavenumber.
Each has a standalone inverse: forward -> filter coefficients -> inverse.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from uacpy.core.constants import (REFERENCE_PRESSURE_AIR,
                                  REFERENCE_PRESSURE_WATER)
from uacpy.core.exceptions import ConfigurationError


_RADON_KINDS = ("linear", "parabolic", "hyperbolic")


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


def radon_transform(data, fs, dx, moveout, kind="linear", x0=0.0):
    """Forward Radon transform (slant stack) of a ``(nt, nx)`` gather.

    Sums the data along moveout curves ``t = t(tau, x)``:

    * ``linear``      ``t = tau + p*x``        (``moveout`` = slowness p, s/m)
    * ``parabolic``   ``t = tau + q*x**2``     (``moveout`` = curvature q, s/m^2)
    * ``hyperbolic``  ``t = sqrt(tau^2+(x/v)^2)`` (``moveout`` = velocity v, m/s)

    Parameters
    ----------
    data : ndarray
        Gather, shape ``(nt, nx)`` (time down columns, offset across rows).
    fs : float
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
    moveout : ndarray
        The scanned moveout axis.
    taus : ndarray
        Intercept-time axis (s).
    R : ndarray
        Radon panel, shape ``(len(moveout), nt)``.
    """
    d = np.asarray(data, dtype=float)
    if d.ndim != 2:
        raise ConfigurationError("radon_transform: data must be 2-D (nt, nx)")
    nt, nx = d.shape
    x = np.arange(nx) * float(dx) - float(x0)
    moveout = np.atleast_1d(np.asarray(moveout, dtype=float))
    taus = np.arange(nt) / float(fs)
    R = np.zeros((moveout.size, nt))
    for i, m in enumerate(moveout):
        for ix in range(nx):
            tt = _moveout_times(kind, taus, x[ix], m)
            R[i] += np.interp(tt, taus, d[:, ix], left=0.0, right=0.0)
    return moveout, taus, R


def inverse_radon(R, fs, dx, moveout, nx, kind="linear", x0=0.0):
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
    taus = np.arange(nt) / float(fs)
    x = np.arange(int(nx)) * float(dx) - float(x0)
    out = np.zeros((nt, int(nx)))
    for i, m in enumerate(moveout):
        for ix in range(int(nx)):
            tt = _moveout_times(kind, taus, x[ix], m)
            out[:, ix] += np.interp(taus, tt, R[i], left=0.0, right=0.0)
    return out


def taup_transform(data, fs, dx, slownesses=None, n_slowness=201, p_max=None):
    """Forward linear tau-p (slant stack), frequency-domain.

    Returns ``(slownesses, taus, taup)``: slowness axis (s/m), intercept-time
    axis (s), and the panel ``(n_slowness, nt)``. See :class:`TauP` for the
    convention; this is the standalone functional form (no stored state).
    """
    d = np.asarray(data, dtype=float)
    if d.ndim != 2:
        raise ConfigurationError("taup_transform: data must be 2-D (nt, nx)")
    nt, nx = d.shape
    fs = float(fs)
    x = np.arange(nx) * float(dx)
    if slownesses is None:
        if p_max is None:
            p_max = 1.0 / 1000.0
        slownesses = np.linspace(-p_max, p_max, int(n_slowness))
    slownesses = np.atleast_1d(np.asarray(slownesses, dtype=float))
    D = np.fft.rfft(d, axis=0)
    omega = 2.0 * np.pi * np.fft.rfftfreq(nt, 1.0 / fs)
    taup = np.empty((slownesses.size, nt))
    for i, p in enumerate(slownesses):
        phase = np.exp(1j * omega[:, None] * (p * x)[None, :])
        taup[i] = np.fft.irfft(np.sum(D * phase, axis=1), n=nt)
    return slownesses, np.arange(nt) / fs, taup


def inverse_taup(taup, slownesses, fs, dx, nx):
    """Adjoint slant stack ``(n_slowness, nt) -> (nt, nx)``.

    Standalone inverse — pass a tau-p panel you already have (e.g. a filtered
    one) plus its slowness axis and geometry; no forward ``compute`` needed.
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
    omega = 2.0 * np.pi * np.fft.rfftfreq(nt, 1.0 / float(fs))
    D = np.zeros((omega.size, int(nx)), dtype=complex)
    for i, p in enumerate(slownesses):
        D += U[i][:, None] * np.exp(-1j * omega[:, None] * (p * x)[None, :])
    return np.fft.irfft(D, n=nt, axis=0)


class Radon:
    """Radon transform (compute + plot), mirroring TauP / FK.

    Stateful wrapper over :func:`radon_transform`: ``compute`` stores the panel
    and ``plot`` images ``|R|`` (modulus taken on a copy, so the stored panel is
    untouched). Reconstruct a (filtered) panel with the standalone
    :func:`inverse_radon`.
    """

    _AXIS = {
        "linear": ("Slowness p [s/km]", 1e3),
        "parabolic": ("Curvature q [s/km^2]", 1e6),
        "hyperbolic": ("Velocity v [m/s]", 1.0),
    }

    def __init__(self, ref: float = 1e-6):
        self.ref = float(ref)

    def compute(self, data, fs, dx, moveout, kind="linear", x0=0.0):
        """Forward Radon transform; see :func:`radon_transform`."""
        d = np.asarray(data, dtype=float)
        if d.ndim != 2:
            raise ConfigurationError("Radon.compute: data must be 2-D (nt, nx)")
        self.moveout, self.taus, self.R = radon_transform(d, fs, dx, moveout,
                                                          kind, x0)
        self.fs, self.dx, self.nx = float(fs), float(dx), d.shape[1]
        self.kind, self.x0 = kind, float(x0)
        return self.moveout, self.taus, self.R

    def plot(self, title="", vmin=None, vmax=None, cmap="jet", **kwargs):
        """Image ``|R|`` (moveout axis on x, intercept time on y).

        To reconstruct a (filtered) panel, pass it to :func:`inverse_radon`.
        """
        if not hasattr(self, "R"):
            raise RuntimeError("Radon.plot: compute() must be called before plotting")
        amp = np.abs(self.R)
        xlabel, scale = self._AXIS.get(self.kind, ("Moveout", 1.0))
        m = self.moveout * scale
        if vmax is None:
            vmax = amp.max()
        if vmin is None:
            vmin = 0.0
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(amp.T, aspect="auto", origin="upper",
                       extent=[m[0], m[-1], self.taus[-1], self.taus[0]],
                       vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        ax.set_title(f"[radon:{self.kind}] {title}", loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Intercept time tau [s]")
        fig.colorbar(im, ax=ax, label="Stack amplitude")
        plt.tight_layout()
        return fig, ax


class TauP:
    """Linear tau-p (slant-stack) transform — time-domain dual of FK.

    Computed in the frequency domain: a time shift ``p*x`` per trace is a phase
    ``exp(i*omega*p*x)``, so ``U(p, omega) = sum_x D(omega, x) exp(i*omega*p*x)``
    and ``u(p, tau)`` is the inverse transform over ``omega``.
    """

    def __init__(self, ref: float = 1e-6):
        self.ref = float(ref)

    def compute(self, data, fs, dx, slownesses=None, n_slowness: int = 201,
                p_max=None):
        """Forward tau-p transform.

        Parameters
        ----------
        data : ndarray
            Gather ``(nt, nx)``.
        fs : float
            Temporal sample rate (Hz).
        dx : float
            Sensor spacing (m).
        slownesses : array, optional
            Slowness axis (s/m). Default: ``n_slowness`` points spanning
            ``+/- p_max`` with ``p_max = 1/1000`` s/m (apparent speeds >= 1000 m/s).
        n_slowness : int
            Number of slownesses when ``slownesses`` is None.
        p_max : float, optional
            Half-width of the default slowness axis (s/m).

        Returns
        -------
        slownesses, taus, taup : ndarray
            Slowness axis (s/m), intercept-time axis (s), and panel
            ``(n_slowness, nt)``.
        """
        d = np.asarray(data, dtype=float)
        if d.ndim != 2:
            raise ConfigurationError("TauP.compute: data must be 2-D (nt, nx)")
        self.slownesses, self.taus, self.taup = taup_transform(
            d, fs, dx, slownesses, n_slowness, p_max)
        self.fs, self.dx, self.nx = float(fs), float(dx), d.shape[1]
        return self.slownesses, self.taus, self.taup

    @staticmethod
    def draw_slowness_line(ax, tau_max, sound_speed, *, color="w", ls="--",
                           lw=1.1, alpha=0.85, label=True):
        """Mark the slowness ``p = 1/c`` of a reference speed as vertical lines.

        In tau-p, a wave of apparent speed ``c`` focuses at ``p = +/- 1/c`` (here
        drawn in s/km). ``sound_speed`` is required — no default line.
        """
        p_skm = 1000.0 / float(sound_speed)
        for sgn in (-1.0, 1.0):
            ax.axvline(sgn * p_skm, color=color, ls=ls, lw=lw, alpha=alpha)
        if label:
            ax.text(p_skm, tau_max, f" {sound_speed:.0f} m/s", color=color,
                    fontsize=8, va="bottom", ha="left")

    def plot(self, title="", vmin=None, vmax=None, sound_speed=None, cmap="jet",
             **kwargs):
        """Image the tau-p panel (slowness in s/km on x, intercept time on y).

        To reconstruct a (filtered) panel, pass it to :func:`inverse_taup`.
        """
        if not hasattr(self, "taup"):
            raise RuntimeError("TauP.plot: compute() must be called before plotting")
        p_skm = self.slownesses * 1000.0
        amp = np.abs(self.taup)
        if vmax is None:
            vmax = amp.max()
        if vmin is None:
            vmin = 0.0
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(amp.T, aspect="auto", origin="upper",
                       extent=[p_skm[0], p_skm[-1], self.taus[-1], self.taus[0]],
                       vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        if sound_speed is not None:
            self.draw_slowness_line(ax, self.taus[-1], sound_speed)
        ax.set_title(f"[tau-p] {title}", loc="left")
        ax.set_xlabel("Slowness p [s/km]")
        ax.set_ylabel("Intercept time tau [s]")
        fig.colorbar(im, ax=ax, label="Stack amplitude")
        plt.tight_layout()
        return fig, ax


def inverse_fk(FK):
    """Inverse f-k transform: complex (fftshifted) spectrum -> real ``(nt, nx)``.

    Pass the (possibly filtered/muted) complex spectrum — e.g. ``FK``'s
    ``.FK`` attribute after applying an f-k mask. ``FK`` must be in the
    ``fftshift``ed layout that ``FK.compute`` produces.
    """
    fk = np.asarray(FK)
    if fk.ndim != 2:
        raise ConfigurationError("inverse_fk: FK must be 2-D (nt, nx)")
    return np.real(np.fft.ifft2(np.fft.ifftshift(fk, axes=(0, 1))))



class FK:
    """Frequency-Wavenumber (f-k) transform computation and visualization."""

    def __init__(self, ref=REFERENCE_PRESSURE_WATER, **kwargs):
        """
        Frequency-Wavenumber (f-k) transform computation and visualization class.

        Parameters
        ----------
        ref : float
            Reference level for dB scaling.
        **kwargs
            Additional keyword arguments.
        """
        self.ref = ref

    @staticmethod
    def draw_sound_cone(ax, f_max, k_max, sound_speed, *, color="w",
                        ls="--", lw=1.1, alpha=0.85, label=True):
        """Overlay the acoustic cone ``f = c * nu`` onto an f-k axis.

        On an f-k plot (x = spatial frequency ``nu`` in cycles/m, y = frequency
        in Hz) a wave with apparent horizontal speed ``c`` lies on the line
        ``f = c * nu``. The cone (the ``V`` of the two mirror lines through the
        origin) bounds the propagating region: acoustic energy has apparent
        speed ``>= c``, so it falls on or inside the cone. ``sound_speed`` (m/s)
        is required — there is no default cone. Used by :meth:`plot` via its
        ``sound_speed`` argument; exposed so it can also annotate an externally
        drawn f-k axis.
        """
        c = float(sound_speed)
        nu = min(f_max / c, k_max)
        f = nu * c
        ax.plot([0, nu], [0, f], color=color, ls=ls, lw=lw, alpha=alpha)
        ax.plot([0, -nu], [0, f], color=color, ls=ls, lw=lw, alpha=alpha)
        if label:
            ax.text(nu, f, f" {c:.0f} m/s", color=color, fontsize=8,
                    va="top", ha="right")

    def compute(self, data, fs, dx, normalize=False):
        """
        Compute the frequency-wavenumber (f-k) spectrum using a 2D FFT.

        Parameters
        ----------
        data : array_like
            2D array with shape (nt, nx).
        fs : float
            Temporal sampling frequency (Hz).
        dx : float
            Spatial sampling interval (m).
        normalize : bool
            If ``False`` (default) ``fk`` is the raw ``|FFT2|^2`` — a *relative*
            power (plotted as relative dB). If ``True`` it is the calibrated
            two-sided 2-D power spectral density
            ``|FFT2|^2 * dx / (fs * nt * nx)``, in ``input_unit^2 / (Hz·cyc/m)``
            — Parseval-consistent (``sum(fk)*df*dk == mean(data^2)``). Assumes a
            rectangular window; for a tapered ``data`` divide by ``mean(w^2)``
            for absolute accuracy.

        Returns
        -------
        freqs : ndarray
            Frequency axis (Hz).
        wavenumbers : ndarray
            Spatial-frequency axis in cycles/m (``= 1/wavelength = k/(2*pi)``,
            from ``np.fft.fftfreq``). This is **not** the angular wavenumber
            ``k = 2*pi/lambda`` (rad/m) used elsewhere in ocean acoustics —
            multiply by ``2*pi`` for that. Phase speed reads off directly as
            ``c = f / wavenumber`` either way.
        fk : ndarray
            2-D power panel (relative ``|FK|^2`` or, with ``normalize``, the
            calibrated PSD). The complex spectrum is kept on ``.FK`` for
            :func:`inverse_fk` and is unaffected by ``normalize``.
        """
        data = np.asarray(data)
        nt, nx = data.shape

        # Forward 2D FFT (complex, kept raw for the inverse)
        FK = np.fft.fftshift(np.fft.fft2(data), axes=(0, 1))

        FKp = np.abs(FK) ** 2
        if normalize:
            # Two-sided 2-D PSD; Parseval: sum(FKp)*df*dk == mean(data**2).
            FKp = FKp * (float(dx) / (float(fs) * nt * nx))

        # Axes
        freqs = np.fft.fftshift(np.fft.fftfreq(nt, d=1 / fs))
        wavenumbers = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))

        # Store results
        self.frequencies = freqs
        self.wavenumbers = wavenumbers
        self.FK = FK
        self.fk = FKp
        self.normalized = bool(normalize)

        return freqs, wavenumbers, FKp

    def plot(self, title="", vmin=-60, vmax=20, sound_speed=None, **kwargs):
        """
        Plot the computed f-k spectrum as an image.

        Parameters
        ----------
        title : str
            Plot title.
        vmin, vmax : float
            Color scale limits (dB).
        sound_speed : float, optional
            If given (m/s), overlay the acoustic cone ``f = sound_speed * nu``.
            Default ``None`` draws no cone.
        **kwargs
            Additional keyword arguments passed to ``ax.imshow``.
        """
        if not hasattr(self, "frequencies") or not hasattr(self, "wavenumbers") or not hasattr(self, "fk"):
            raise RuntimeError(
                "FK.plot: compute() must be called before plotting"
            )

        floor = np.finfo(float).tiny
        fk_db = 10 * np.log10(np.maximum(self.fk, floor) / (self.ref ** 2))

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            fk_db,
            extent=[
                self.wavenumbers[0],
                self.wavenumbers[-1],
                self.frequencies[0],
                self.frequencies[-1],
            ],
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

        ax.set_title(f"[f–k] {title}", loc="left")
        ax.set_xlabel("Spatial frequency [cycles/m]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlim((self.wavenumbers[0], self.wavenumbers[-1]))
        ax.set_ylim((0, self.frequencies[-1]))
        ax.grid(alpha=0.3)

        if sound_speed is not None:
            self.draw_sound_cone(ax, self.frequencies[-1],
                                 self.wavenumbers[-1], sound_speed)

        cbar = fig.colorbar(im, ax=ax)
        if getattr(self, "normalized", False):
            if self.ref == REFERENCE_PRESSURE_WATER:
                ref = "1µ"
            elif self.ref == REFERENCE_PRESSURE_AIR:
                ref = "20µ"
            else:
                ref = f"{self.ref:.0e} "
            cbar.set_label(f"PSD [dB re {ref}Pa²/(Hz·cyc/m)]")
        else:
            # raw |FFT2|^2: relative (uncalibrated) power, not a spectral density
            cbar.set_label("Relative power [dB]")

        plt.tight_layout()
        return fig, ax
