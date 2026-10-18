"""The discrete-time Fourier transform of a record at one frequency.

:func:`dtft_at` is ``Σ_n x_n w_n exp(-2πi f t_n)`` along one axis, on any
sample times, without transforming the other frequencies.
:func:`uacpy.acoustics.snapshot_frequency_component` and
:func:`uacpy.acoustic_signal.tone_phasor` are this sum with their own gains,
so the transform, its phase origin and its memory bound are one
implementation.

This module sits in ``core`` because ``acoustic_signal`` and
``core.acoustics`` both call it and ``core`` must depend on neither.
"""

import numpy as np

#: Complex128 elements of cast record per block on the windowed path
#: (~64 MB), the scratch budget the synthesis chunking in
#: ``acoustic_signal/_synthesis.py`` (``synthesize_cells``) works to.
_BLOCK_ELEMENTS = 4_000_000


def dtft_at(data, times, frequency: float, *, window=None, axis: int = 0):
    """``Σ_n x_n w_n exp(-2πi f t_n)`` along ``axis``: the transform of
    ``data`` evaluated at ``frequency``, with ``t_n`` the time of sample
    ``n`` (any spacing; the phase is referenced to ``t = 0``) and ``w_n``
    the real per-sample ``window`` (``None``: all ones).

    On a uniform grid ``t_n = n·dt`` this is row ``f·nt·dt`` of
    ``np.fft.fft(data, axis=axis)``, and a fractional row is the transform
    between the DFT bins. ``np.fft.fft(cube, axis=0)[k]`` transforms all
    ``nt`` frequencies and keeps one — measured 5.1x the cube in peak RSS,
    which is ~17 GB on a 512 x 200 x 4096 snapshot, to produce an
    (nrd, nk) slab. That row is a contraction of ``data`` against
    ``exp(-2i pi f t_n)``, so it costs the slab plus one pass over the
    record instead.

    Precision. Without ``window`` the result keeps the dtype the full
    transform gives (numpy's FFT rule): single in, single out; anything
    else double. With ``window`` numpy's ``data * window`` promotes a
    complex64 record to complex128, so that FFT runs in DOUBLE precision
    and is accurate to ~1e-16 rather than the ~1e-7 of a single-precision
    transform. Contracting a complex128 kernel against the whole record
    would promote it the same way and hand most of the memory back (3.2x
    the cube against 1.6x), so the windowed path casts and accumulates in
    complex128 over blocks of the last other axis: same accuracy, bounded
    scratch.

    Parameters
    ----------
    data : array_like
        The record, time along ``axis``.
    times : array_like
        Time of each sample along ``axis`` (s, or samples), one per sample.
    frequency : float
        The frequency, in the reciprocal unit of ``times``.
    window : array_like, optional
        Real taper, one weight per sample.
    axis : int, default 0
        Time axis; the result has ``data``'s shape without it.

    Returns
    -------
    ndarray or complex
        The sum, ``data``'s shape without ``axis``.
    """
    x = np.moveaxis(np.asarray(data), axis, 0)
    t = np.asarray(times, dtype=float).ravel()
    phase = np.exp(-2j * np.pi * float(frequency) * t)
    if window is None:
        # Mirror numpy's own FFT precision rule so the row keeps the dtype the
        # full transform gives it: single in, single out; anything else double.
        single = np.result_type(x.dtype, np.complex64) == np.complex64
        return np.tensordot(
            phase.astype(np.complex64 if single else np.complex128), x, axes=1)
    kernel = phase * np.asarray(window, dtype=np.float64).ravel()
    if x.ndim == 1:
        return np.dot(kernel, x.astype(np.complex128))
    out = np.empty(x.shape[1:], dtype=np.complex128)
    per_column = max(int(np.prod(x.shape[:-1])), 1)
    step = max(1, _BLOCK_ELEMENTS // per_column)
    for a in range(0, x.shape[-1], step):
        out[..., a:a + step] = np.tensordot(
            kernel, x[..., a:a + step].astype(np.complex128), axes=1)
    return out
