"""The shared frequency-domain equaliser denominator.

Three consumers divide by ``|H|^2`` and need the same offset to keep a
spectral null finite:

* :func:`uacpy.comms.ofdm.ofdm_demodulate` — per-subcarrier, over one OFDM
  block's channel response;
* :func:`uacpy.comms.equalization.mmse_equalizer` — circular, over the
  full-record DFT of a single-carrier channel;
* :meth:`uacpy.comms.transceiver.OFDMReceiver.equalize` — the same
  per-subcarrier division inside the receiver object.

The offset is one formula in the units of ``|H|^2``, so it lives here rather
than inside whichever consumer happened to need it first; a consumer that
changes its own equaliser policy (which scheme, which SNR estimate) changes
its call, not this.

This module holds no I/O and no scheme knowledge, so all three can import it
without importing each other.
"""

# Zero-forcing floor, as a fraction of the channel's own peak power |H|^2.
# Relative because |H|^2 carries whatever amplitude scale the caller's channel
# is in: an absolute floor silently makes the result a function of those units
# — a channel holding 1e-5 of propagation gain equalised to EVM 0.014, and
# 1e-6 to 0.54. `system_id._etfe_divide` states the same rule for a transfer
# function, and `ofdm._PILOT_REL_FLOOR` for a pilot magnitude.
_ZF_REL_FLOOR = 1e-12


def regularizer(h2, snr_linear):
    """Denominator offset for ``conj(H)/(|H|^2 + eps)``, in the units of ``h2``.

    Zero-forcing (``snr_linear is None``) uses a floor at ``_ZF_REL_FLOOR`` of
    the peak subcarrier power, which only keeps a spectral null finite. MMSE
    uses the physical noise-to-signal ratio, expressed in those same units:
    ``snr_linear`` is the SNR at the equalizer input, so the noise power that
    goes with it is ``mean(|H|^2) / snr``. Both return 0.0 for a channel with
    no power at all, which leaves the caller's ``0/0`` to be masked there.
    """
    peak = float(h2.max()) if h2.size else 0.0
    if peak <= 0.0:
        return 0.0
    if snr_linear is None:
        return _ZF_REL_FLOOR * peak
    return float(h2.mean()) / float(snr_linear)
