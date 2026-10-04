"""The PE field resampled onto the receiver grid: the interpolation of the
complex envelope and of the TL."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interp_envelope_to_receiver_grid(src_depths, src_ranges, psi,
                                      rcv_depths, rcv_ranges, *,
                                      carrier_rate=0.0):
    """Resample a complex PE envelope, interpolating modulus and phase apart.

    The Collins output grid is spaced ``dr·ndr`` — sized for Padé march
    accuracy, with nothing tying it to the envelope's range Nyquist. When
    ``c0`` sits far from the water-column speed, ψ still rotates fast enough
    that adjacent output samples are >90° apart, and interpolating the
    complex field averages across opposite-phase lobes: on the Pekeris
    reference (100 m, c=1500, half-space 1700/1.7/0.5, 250 Hz) that raises
    the median TL by 1.5-2.3 dB.

    The modulus has no carrier — it varies on the modal-beat scale — so it
    is well sampled and is interpolated on its own, tracking the binary's
    own ``tl.grid`` resampling to ~0.05 dB median. The unit phasor is
    interpolated separately and renormalised, so it carries phase only and
    contributes nothing to the level.

    ``carrier_rate`` (rad/m, see :func:`_stability.collins_carrier_rate`) is
    the phase rate the backend baked into ψ. Linear interpolation of the
    unit phasor is only faithful while the rotation between two source
    samples stays under π, so the carrier is divided out on the source grid,
    the slowly varying residual is interpolated, and the carrier is restored
    on the receiver grid — the two are exact inverses at the source samples,
    so the grid values are untouched."""
    psi = np.asarray(psi)
    src_r = np.asarray(src_ranges, dtype=float)
    rcv_r = np.atleast_1d(np.asarray(rcv_ranges, dtype=float))
    if carrier_rate:
        psi = psi * np.exp(-1j * float(carrier_rate) * src_r)[None, :]
    mag = np.abs(psi)
    with np.errstate(invalid='ignore', divide='ignore'):
        unit = np.where(mag > 0.0, psi / mag, 0.0)
    mag_out = interp_to_receiver_grid(
        src_depths, src_r, mag, rcv_depths, rcv_r)
    unit_out = interp_to_receiver_grid(
        src_depths, src_r, unit, rcv_depths, rcv_r)
    with np.errstate(invalid='ignore', divide='ignore'):
        mod = np.abs(unit_out)
        unit_out = np.where(mod > 0.0, unit_out / mod, 1.0 + 0.0j)
    out = mag_out * unit_out
    if carrier_rate:
        out = out * np.exp(1j * float(carrier_rate) * rcv_r)[None, :]
    return out


def interp_to_receiver_grid(src_depths, src_ranges, values,
                             rcv_depths, rcv_ranges):
    """Bilinear-interpolate a PE-grid ``(depth, range)`` field onto the
    receiver grid. Real or complex ``values`` (complex via independent re/im);
    out-of-grid samples become NaN.

    A NaN in the source grid propagates into every receiver cell whose
    stencil touches it. That is the point: a sample the march failed to
    solve is no data, and substituting a zero for it would make it a real
    pressure the interpolator averages against its neighbours, so cells
    around a failure would come back finite and credible.

    Returns ``(len(rcv_depths), len(rcv_ranges))``."""
    rd = np.atleast_1d(np.asarray(rcv_depths, dtype=float))
    rr = np.atleast_1d(np.asarray(rcv_ranges, dtype=float))
    grid = (np.asarray(src_depths, dtype=float), np.asarray(src_ranges, dtype=float))
    DD, RR = np.meshgrid(rd, rr, indexing='ij')
    pts = np.stack([DD.ravel(), RR.ravel()], axis=-1)

    def _one(v):
        v = np.asarray(v)
        rgi = RegularGridInterpolator(grid, v.astype(np.float64),
                                      bounds_error=False, fill_value=np.nan)
        return rgi(pts).reshape(DD.shape)

    if np.iscomplexobj(values):
        return _one(np.real(values)) + 1j * _one(np.imag(values))
    return _one(values)
