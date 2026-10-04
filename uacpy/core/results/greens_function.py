"""Depth-separated Green's function result type."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from uacpy.core.acoustics.wavenumber import (
    _warn_zero_ranges, hankel_transform, snapshot_frequency_component,
    wavenumber_taper, wavenumbers_from_phase_speeds,
)
from uacpy.core.exceptions import ConfigurationError, ProvenanceWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP

from uacpy.core.results._base import Result, coordinate_axis
from uacpy.core.results.quantities import coordinate_unit
from uacpy.core.results.field import Field


class GreensFunction(Result):
    """Depth-separated Green's function ``G(k, z)`` of a wavenumber-integration run.

    What SCOOTER, and SPARC in snapshot mode, write to a ``.grn`` before the
    transform to range: the response of the stratified column to a source
    driven at horizontal wavenumber ``k`` (JKPS Sect. 4).
    :func:`uacpy.io.read_grn_file` returns one. The methods transform it to
    a range-domain :class:`Field` through the functions of
    :mod:`uacpy.core.acoustics.wavenumber`, which take plain arrays.

    The grid is stored as phase speeds, not wavenumbers; :meth:`wavenumbers`
    gives ``k = 2*pi*f / c`` at a frequency.

    Attributes
    ----------
    data : ndarray, shape ``(n_slots, n_source_depths, n_receiver_depths, n_k)``
        The complex Green's function. ``n_slots`` is the frequency axis of a
        SCOOTER file and the output-time axis of a SPARC snapshot.
    phase_speeds : ndarray, shape ``(n_k,)``
        The phase-speed grid (m/s) the kernel was sampled on.
    receiver_depths : ndarray, shape ``(n_receiver_depths,)``
        Metres.
    frequencies : ndarray
        Hz. A SCOOTER file's frequency axis; a SPARC snapshot's one source
        frequency, at which its wavenumber grid is fixed.
    times : ndarray or None
        Seconds. A SPARC snapshot's output times (``sparc.f90:320`` stores
        them in the file's frequency slot); ``None`` for a SCOOTER file.
    stabilizing_attenuation : float
        The contour offset ``Atten`` the file's header records (1/m). The
        solver evaluates the kernel on ``k + i*Atten`` (``scooter.f90:581``);
        the transforms undo it.
    title : str
        The file's title line.
    """

    def __init__(
        self,
        *,
        data: np.ndarray,
        phase_speeds: np.ndarray,
        receiver_depths: np.ndarray,
        times: Optional[np.ndarray] = None,
        stabilizing_attenuation: float = 0.0,
        title: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Not copied: a snapshot cube reaches tens of GB.
        self.data = np.asarray(data)
        self.phase_speeds = np.asarray(phase_speeds, dtype=float)
        # Kept at the precision given: the file stores the depths in float32,
        # and the fields built from this carry them as their depth axis.
        self.receiver_depths = np.atleast_1d(np.asarray(receiver_depths))
        self.times = None if times is None else np.asarray(times, dtype=float)
        self.stabilizing_attenuation = float(stabilizing_attenuation)
        self.title = str(title)
        if self.data.ndim != 4:
            raise ConfigurationError(
                f"GreensFunction: data must be 4-D (n_slots, "
                f"n_source_depths, n_receiver_depths, n_k); got shape "
                f"{self.data.shape}.")
        n_slots = (len(self.times) if self.is_snapshot
                   else self.n_frequencies)
        expected = (n_slots, len(self.source_depths),
                    len(self.receiver_depths), len(self.phase_speeds))
        if self.data.shape != expected:
            raise ConfigurationError(
                f"GreensFunction: data has shape {self.data.shape}, but "
                f"the axes give {expected} ("
                f"{'times' if self.is_snapshot else 'frequencies'}, "
                f"source_depths, receiver_depths, phase_speeds).")
        if self.is_snapshot and self.n_frequencies != 1:
            raise ConfigurationError(
                f"GreensFunction: a snapshot (times given) carries its one "
                f"source frequency; got {self.n_frequencies} frequencies.")

    @property
    def is_snapshot(self) -> bool:
        """Whether this is a SPARC snapshot: its first axis holds output
        times, and its wavenumber grid is fixed at the source frequency."""
        return self.times is not None

    def _repr_bits(self) -> list:
        bits = [coordinate_axis('receiver_depth', self.receiver_depths),
                coordinate_axis('phase_speed', self.phase_speeds)]
        if self.is_snapshot:
            bits.append(f"snapshot {coordinate_axis('time', self.times)}")
        return bits

    # ── axes ───────────────────────────────────────────────────────────

    def wavenumbers(self, frequency: Optional[float] = None) -> np.ndarray:
        """The horizontal-wavenumber axis (rad/m) at ``frequency``.

        :func:`~uacpy.core.acoustics.wavenumbers_from_phase_speeds` on the
        stored grid. SCOOTER recomputes ``k`` per frequency from one
        phase-speed grid (``scooter.f90:127``), so ``frequency`` selects the
        axis; it may be omitted for a single-frequency file. A snapshot's
        grid is fixed at its source frequency and takes no ``frequency``.

        Parameters
        ----------
        frequency : float, optional
            Frequency (Hz) of the axis; may be omitted for a single-frequency
            file.

        Returns
        -------
        ndarray, shape ``(n_k,)``
            In the order of :attr:`phase_speeds`.
        """
        if self.is_snapshot:
            if frequency is not None:
                raise ConfigurationError(
                    f"GreensFunction.wavenumbers: a snapshot's wavenumber "
                    f"grid is fixed at its source frequency "
                    f"{self.f0:g} Hz; call it without frequency=.")
            return wavenumbers_from_phase_speeds(self.phase_speeds, self.f0)
        if frequency is None:
            if self.n_frequencies != 1:
                raise ConfigurationError(
                    f"GreensFunction.wavenumbers: the Green's function holds "
                    f"{self.n_frequencies} frequencies; say which one.")
            frequency = self.f0
        return wavenumbers_from_phase_speeds(self.phase_speeds, frequency)

    def _contour_attenuation(self, k: np.ndarray) -> float:
        """The stabilising attenuation the transform undoes at grid ``k``.

        SCOOTER evaluates the FE solve on the contour ``k + i*Atten``
        (``scooter.f90:581``), so the inverse transform has to undo that
        offset with ``exp(+Atten*r)`` — which means using the ``Atten`` the
        solver actually used.

        ``scooter.f90:122-125`` recomputes ``Deltak = (kMax - kMin)/(Nk - 1)``
        inside the frequency loop from ``kMin = omega/cHigh``, so ``Deltak`` —
        and hence ``Atten`` — scales with frequency, while
        ``scooter.f90:133`` writes the header only for ``ifreq == 1``. That
        is why ``fieldsco.m:113-115`` re-derives ``Atten`` from the file's
        own per-frequency ``k`` vector instead of reading the header, and it
        is right to for a broadband run.

        But ``scooter.f90:130`` zeroes ``Atten`` for every frequency when
        ``TopOpt(7:7) == '0'``, and a zero header is therefore valid at every
        frequency. Re-deriving ``Δk`` there multiplies a Green's function
        computed on the real axis by ``exp(+Δk*r)``, biasing TL by
        ``8.686*Δk*r`` dB — 6.8 dB at the far receiver on the default
        ``rmax_factor = 2.0``. ``fieldsco.m`` has no way to know the flag
        was set; uacpy wrote it, and the value the solver used is in the
        header, so it is used.

        A snapshot gives 0 (``sparc.f90:313``). For other titles the header
        is taken as given.
        """
        if self.is_snapshot:
            return 0.0
        header_atten = self.stabilizing_attenuation
        if (self.title.upper().startswith('SCOOTER') and header_atten != 0.0
                and len(k) > 1):
            return float(k[1] - k[0])
        return header_atten

    def _check_source_depth_idx(self, source_depth_idx: int) -> None:
        nsd = len(self.source_depths)
        if not (0 <= source_depth_idx < nsd):
            raise ConfigurationError(
                f"source_depth_idx={source_depth_idx} out of range for "
                f"nsd={nsd}."
            )

    def _refuse_snapshot(self, method: str) -> None:
        if self.is_snapshot:
            raise ConfigurationError(
                f"GreensFunction.{method} transforms a frequency-domain "
                f"Green's function; this is a SPARC snapshot (title "
                f"{self.title!r}), whose first axis holds output TIMES "
                f"(sparc.f90:317-319).",
                remediation="Use snapshot_to_field for the steady-state "
                            "field at one frequency, or "
                            "snapshot_to_time_field for p(z, r, t).")

    def _refuse_frequency_domain(self, method: str) -> None:
        if not self.is_snapshot:
            raise ConfigurationError(
                f"GreensFunction.{method} expects a SPARC snapshot; got "
                f"title {self.title!r} (no output times).",
                remediation="Use to_field or to_transfer_function for a "
                            "frequency-domain Green's function.")

    def _pressure_slice(self, ranges, ifreq, isd, *, source_type, spectrum,
                        cmin, cmax) -> np.ndarray:
        """Transform one (frequency, source depth) slab to the range domain."""
        G_src = self.data[ifreq, isd, :, :]                     # (nrd, nk)
        freq_i = float(self.frequencies[ifreq])
        k = self.wavenumbers(freq_i)
        atten = self._contour_attenuation(k)
        if cmin is not None or cmax is not None:
            win = wavenumber_taper(k, freq_i, cmin, cmax)
            G_src = G_src * win[np.newaxis, :]
        return hankel_transform(
            G_src, k, ranges,
            attenuation=atten,
            source_type=source_type,
            spectrum=spectrum,
        )

    # ── frequency-domain transforms ────────────────────────────────────

    def to_field(
        self,
        ranges: np.ndarray,
        *,
        frequency: Optional[float] = None,
        source_type: str = 'point',
        spectrum: str = 'positive',
        source_depth_idx: int = 0,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
    ) -> Field:
        """Transform one frequency to a complex narrowband :class:`Field`.

        :func:`~uacpy.core.acoustics.hankel_transform` of one slab, after the
        optional :func:`~uacpy.core.acoustics.wavenumber_taper`. A
        single-frequency file is transformed at its one slice. A
        multi-frequency file needs ``frequency=``, which selects the nearest
        stored frequency; without it the call raises rather than pick one
        (:meth:`to_transfer_function` transforms every frequency at once).

        The level is the Hankel transform's own: a point source (``'point'``) is
        at the package's unit-source-at-1 m level, while a line source
        (``'line'``) returns ``1/√(k0·r)`` in free space — the ``Scooter`` model
        multiplies it by ``√k0`` after this call to reach unit amplitude at
        1 m.

        Parameters
        ----------
        ranges : array_like
            Receiver ranges (m).
        frequency : float, optional
            Hz. Selects the nearest frequency slice of a multi-frequency
            file; required there. The returned field is labelled with the
            slice's own frequency.
        source_type, spectrum : see :func:`~uacpy.core.acoustics.hankel_transform`.
        source_depth_idx : int
            Index into the source-depth axis. Defaults to the first source.
        cmin, cmax : float, optional
            Phase-speed taper bounds (m/s).

        Raises
        ------
        ConfigurationError
            A snapshot, a multi-frequency file without ``frequency=``, or
            ``source_depth_idx`` out of range.
        """
        self._refuse_snapshot('to_field')
        if self.n_frequencies > 1 and frequency is None:
            raise ConfigurationError(
                f"GreensFunction.to_field: the Green's function holds "
                f"{self.n_frequencies} frequencies "
                f"({float(self.frequencies[0]):g} to "
                f"{float(self.frequencies[-1]):g} Hz); "
                f"say which one to transform.",
                remediation="Pass frequency= (the nearest slice is taken), "
                            "or use to_transfer_function for all of them.")
        ifreq = (int(np.argmin(np.abs(self.frequencies - float(frequency))))
                 if self.n_frequencies > 1 else 0)
        self._check_source_depth_idx(source_depth_idx)
        _warn_zero_ranges(ranges, source_type, context=self.title)

        p_out = self._pressure_slice(
            ranges, ifreq, source_depth_idx,
            source_type=source_type, spectrum=spectrum, cmin=cmin, cmax=cmax,
        )
        return Field(
            data=p_out,
            coords={'depth': self.receiver_depths, 'range': ranges},
            model='', backend='',
            # The slab holds one source depth — the one ``source_depth_idx``
            # selected — so it carries that depth alone.
            source_depths=np.atleast_1d(self.source_depths[source_depth_idx]),
            # Labelled with the frequency of the slab transformed.
            frequencies=float(self.frequencies[ifreq]),
            phase_reference='travelling_wave',
            metadata={
                "transform_method": "direct_dft",
                "source_type": source_type,
                "spectrum": spectrum,
            },
        )

    def to_transfer_function(
        self,
        ranges: np.ndarray,
        *,
        source_type: str = 'point',
        spectrum: str = 'positive',
        source_depth_idx: int = 0,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
    ) -> Field:
        """Transform every frequency to a broadband :class:`Field`.

        Output: complex ``Field`` with ``coords={'depth', 'range',
        'frequency'}``, shape ``(n_d, n_r, n_f)``. The level is the Hankel
        transform's own, as for :meth:`to_field`: a line source (``'line'``)
        still needs the ``√k0`` per frequency the ``Scooter`` model applies.

        Parameters
        ----------
        ranges : ndarray
            Receiver ranges (m).
        source_type, spectrum : str, optional
            See :func:`~uacpy.core.acoustics.hankel_transform`. Default
            ``'point'`` and ``'positive'``.
        source_depth_idx : int, optional
            Index into the source-depth axis. Default 0.
        cmin, cmax : float, optional
            Phase-speed taper bounds (m/s).

        Raises
        ------
        ConfigurationError
            A snapshot, whose first axis holds output times, or
            ``source_depth_idx`` out of range.
        """
        self._refuse_snapshot('to_transfer_function')
        self._check_source_depth_idx(source_depth_idx)
        _warn_zero_ranges(ranges, source_type, context=self.title)
        nfreq = self.n_frequencies
        pressure = np.zeros((len(self.receiver_depths), len(ranges), nfreq),
                            dtype=np.complex128)
        for ifreq in range(nfreq):
            pressure[:, :, ifreq] = self._pressure_slice(
                ranges, ifreq, source_depth_idx,
                source_type=source_type, spectrum=spectrum,
                cmin=cmin, cmax=cmax,
            )
        freqs = self.frequencies.copy()
        return Field(
            data=pressure,
            coords={
                'depth': self.receiver_depths,
                'range': ranges,
                'frequency': freqs,
            },
            phase_reference='travelling_wave',
            model='', backend='',
            # The slab holds one source depth — the one ``source_depth_idx``
            # selected — so it carries that depth alone.
            source_depths=np.atleast_1d(self.source_depths[source_depth_idx]),
            frequencies=freqs,
            metadata={
                'center_frequency': float(freqs[len(freqs) // 2]),
                'nfreq': nfreq,
                'source_type': source_type,
                'spectrum': spectrum,
            },
        )

    # ── snapshot transforms ────────────────────────────────────────────

    def snapshot_to_field(
        self,
        ranges: np.ndarray,
        frequency: float,
        *,
        source_type: str = 'point',
        spectrum: str = 'positive',
        source_depth_idx: int = 0,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        source_waveform: Optional[np.ndarray] = None,
        normalize: str = 'source',
    ) -> Field:
        """Steady-state complex pressure at ``frequency`` from a SPARC snapshot.

        SPARC propagates the *actual* pulse, so the snapshot is ``S(omega)*g``
        — the source spectrum times the transfer function — not the bare
        ``g`` that Kraken/Scooter report (Jensen, *Computational Ocean
        Acoustics*, Eq. 8.1). Two steps:

        1. :func:`~uacpy.core.acoustics.snapshot_frequency_component` along
           the output-time axis, evaluated **at** ``frequency``;
        2. :func:`~uacpy.core.acoustics.hankel_transform` to range.

        ``normalize='source'`` (default) deconvolves ``source_waveform``, the
        pulse SPARC marched sampled at :attr:`times`, recovering ``g`` —
        **absolute TL re 1 m**, directly comparable to Kraken/Scooter. For a
        canned SPARC pulse that is::

            from uacpy.acoustic_signal.generate import sparc_pulse
            waveform, _ = sparc_pulse(gf.times, f, 'P')
            gf.snapshot_to_field(ranges, f, source_waveform=waveform)

        with the pulse letter of the run (``SPARC(pulse_type=...)[0]``). A
        residual remains only from the source high cut ``fHiCut``
        (``sparc.f90:397-401``), which ``sparc_pulse`` does not replicate —
        ``rkT·cHigh/2π`` for ``Pulse(4:4)`` in ``'HB'`` and ``10·fMax``
        otherwise. ``fMin``/``fMax`` themselves are **not** a band-pass on
        the output: they set the wavenumber integration limits
        ``kMin = 2π·fMin/cHigh`` and ``kMax = 2π·fMax/cLow``
        (``sparc.f90:111-112``). ``normalize=None`` returns the raw
        (uncalibrated) field and warns.

        Returns a complex narrowband :class:`Field` (``coords={'depth',
        'range'}``); use ``.dB`` or ``.to_dB()`` for transmission loss in dB.

        Parameters
        ----------
        ranges : ndarray
            Receiver ranges (m).
        frequency : float
            Frequency (Hz) to evaluate at.
        source_type, spectrum : str, optional
            See :func:`~uacpy.core.acoustics.hankel_transform`. Default
            ``'point'`` and ``'positive'``.
        source_depth_idx : int, optional
            Index into the source-depth axis. Default 0.
        cmin, cmax : float, optional
            Phase-speed taper bounds (m/s).
        source_waveform : ndarray, optional
            The pulse SPARC marched, sampled at :attr:`times` (see above).
        normalize : {'source', None}, optional
            Deconvolve ``source_waveform`` (default), or return the raw field.

        Raises
        ------
        ConfigurationError
            A frequency-domain Green's function, an unknown ``normalize``,
            ``normalize='source'`` without ``source_waveform``,
            ``source_depth_idx`` out of range, or a failure of
            :func:`~uacpy.core.acoustics.snapshot_frequency_component`.
        """
        self._refuse_frequency_domain('snapshot_to_field')
        if normalize not in ('source', None):
            raise ConfigurationError(
                "GreensFunction.snapshot_to_field: normalize must be 'source' "
                f"or None; got {normalize!r}.")
        if normalize == 'source' and source_waveform is None:
            raise ConfigurationError(
                "GreensFunction.snapshot_to_field: normalize='source' needs "
                "source_waveform (the pulse SPARC marched, sampled at "
                "self.times) to deconvolve the source spectrum. Pass it, or "
                "use normalize=None for the raw (uncalibrated) field.")
        if normalize is None:
            warnings.warn(
                "GreensFunction.snapshot_to_field: normalize=None returns "
                "the RAW field (S(omega)*g), whose absolute level is "
                "uncalibrated — a pulse-dependent offset (tens of dB) above "
                "calibrated TL (Jensen, Computational Ocean Acoustics, Eq. "
                "8.1). Use normalize='source' (with source_waveform) for "
                "calibrated absolute TL re 1 m, or treat only the field SHAPE "
                "as indicative.",
                ProvenanceWarning, skip_file_prefixes=USER_FRAME_SKIP)
        self._check_source_depth_idx(source_depth_idx)
        _warn_zero_ranges(ranges, source_type, context=self.title)

        G = self.data[:, source_depth_idx, :, :]          # (nt, nrd, nk)
        G_at_f0 = snapshot_frequency_component(
            G, self.times, frequency,
            source_waveform=(source_waveform if normalize == 'source'
                             else None))
        nt = len(self.times)
        dt = float(self.times[1] - self.times[0])
        fft_freqs = np.fft.fftfreq(nt, dt)
        nearest_bin = float(fft_freqs[np.argmin(np.abs(fft_freqs - frequency))])

        # SPARC's k grid does not depend on the frequency asked for.
        k = self.wavenumbers()
        if cmin is not None or cmax is not None:
            taper = wavenumber_taper(k, frequency, cmin, cmax)
            G_at_f0 = G_at_f0 * taper[np.newaxis, :]

        p_out = hankel_transform(
            G_at_f0, k, ranges,
            attenuation=self._contour_attenuation(k),
            source_type=source_type, spectrum=spectrum,
        )
        if normalize is None:
            # Put the RAW field on the full inverse-Hankel weight
            # Δk·√(2k/(πr)) that sparc.f90's 'D' branch carries (:595 kernel
            # with the 1/√(π·Rr) write scale at :292) — the fieldsco-style
            # Hankel above carries 1/√(2πr) and the Scooter −1 prefactor
            # instead, a constant −2 between the two. The calibrated path
            # SKIPS this: after deconvolution G_at_f0 is the Scooter
            # unit-source Green's function, so the bare Hankel already
            # matches Scooter/Kraken.
            p_out = p_out * (-2.0)

        return Field(
            data=p_out,
            coords={'depth': self.receiver_depths, 'range': ranges},
            model='', backend='',
            # The slab holds one source depth — the one ``source_depth_idx``
            # selected — so it carries that depth alone.
            source_depths=np.atleast_1d(self.source_depths[source_depth_idx]),
            frequencies=float(frequency),
            phase_reference='travelling_wave',
            metadata={
                "transform_method": "time_fft+hankel",
                "normalize": normalize,
                "absolute_tl_calibrated": normalize == 'source',
                "snapshot_freq_bin": nearest_bin,
                "snapshot_dt": dt,
                "snapshot_nt": nt,
                "source_type": source_type,
                "spectrum": spectrum,
            },
        )

    def snapshot_to_time_field(
        self,
        ranges: np.ndarray,
        *,
        source_type: str = 'point',
        spectrum: str = 'positive',
        source_depth_idx: int = 0,
    ) -> Field:
        """Range-domain time evolution ``p(z, r, t)`` of a SPARC snapshot.

        ``sparc.f90:580-591`` writes ``Green(Itout, irz, ik)`` — the
        *wavenumber-domain* field at each output time — and
        ``WriteHeaderSparc`` (``:317-327``) stores the time vector in the
        ``.grn``'s frequency slot and the phase-speed vector
        ``sqrt(omega2)/k`` in its range slot. ``doc/sparc.htm`` prescribes
        running FIELDS afterwards "to convert the '.GRN' file to a '.SHD'
        file containing the pressure field"; this is that step done in-tree.

        Simpler than :meth:`snapshot_to_field`, which recovers a *CW*
        component: the snapshot already is the propagated pulse, so one
        :func:`~uacpy.core.acoustics.hankel_transform` per output time gives
        ``p(z, r, t)`` directly — no time-FFT, no frequency selection, and no
        source deconvolution (exactly as the ``'R'``/``'D'`` modes return
        their raw received time series). The wavenumber grid is the one at
        the source frequency, constant across the time axis.

        Returns a real :class:`Field` with ``coords={'depth', 'range',
        'time'}``.

        Parameters
        ----------
        ranges : ndarray
            Receiver ranges (m).
        source_type, spectrum : str, optional
            See :func:`~uacpy.core.acoustics.hankel_transform`. Default
            ``'point'`` and ``'positive'``.
        source_depth_idx : int, optional
            Index into the source-depth axis. Default 0.
        """
        self._refuse_frequency_domain('snapshot_to_time_field')
        self._check_source_depth_idx(source_depth_idx)
        _warn_zero_ranges(ranges, source_type, context=self.title)
        G = self.data[:, source_depth_idx, :, :]          # (nt, nrd, nk)
        tout = self.times
        k = self.wavenumbers()
        atten = self._contour_attenuation(k)                # 0 for SPARC
        ranges = np.atleast_1d(np.asarray(ranges, dtype=float))

        # (nrd, n_ranges) per output time -> (nrd, n_ranges, nt).
        p_t = np.stack(
            [hankel_transform(G[it], k, ranges, attenuation=atten,
                              source_type=source_type, spectrum=spectrum)
             for it in range(G.shape[0])],
            axis=-1,
        )
        # Put the snapshot on the same inverse-Hankel weight the 'D' branch
        # uses, dk*sqrt(2k/(pi*r)) (sparc.f90:595 with the 1/sqrt(pi*Rr)
        # write scale at :292). The fieldsco-style Hankel above carries
        # 1/sqrt(2*pi*r) and the Scooter -1 prefactor instead; the constant
        # between the two is -2.
        p_t = p_t * (-2.0)

        # The snapshot is a real transient field; the Hankel transform
        # carries the analytic-signal convention, so the physical pressure
        # is its real part.
        dt = float(tout[1] - tout[0]) if tout.size > 1 else float('nan')

        return Field(
            data=np.real(p_t),
            coords={'depth': self.receiver_depths, 'range': ranges,
                    'time': tout},
            model='', backend='',
            # The slab holds one source depth — the one ``source_depth_idx``
            # selected — so it carries that depth alone.
            source_depths=np.atleast_1d(self.source_depths[source_depth_idx]),
            frequencies=self.f0,
            phase_reference='time_domain_native',
            metadata={
                "transform_method": "hankel_per_snapshot_time",
                "dt": dt,
                "fs": (1.0 / dt) if dt == dt and dt else float('nan'),
                "nt": int(tout.size),
                "t_start": float(tout[0]) if tout.size else 0.0,
                "source_type": source_type,
                "spectrum": spectrum,
            },
        )

    # ── persistence ────────────────────────────────────────────────────

    #: ``attrs`` the xarray export writes besides the identity.
    _OWN_ATTRS = ('stabilizing_attenuation', 'title')

    def _payload(self):
        slot = 'time' if self.is_snapshot else 'frequency'
        return {'data': (self.data,
                         (slot, 'source_depth', 'receiver_depth',
                          'phase_speed'), '')}

    def _coords(self):
        coords = {}
        if self.is_snapshot:
            coords['time'] = (self.times, coordinate_unit('time'))
        else:
            coords['frequency'] = (self.frequencies,
                                   coordinate_unit('frequency'))
        coords['source_depth'] = (self.source_depths,
                                  coordinate_unit('source_depth'))
        coords['receiver_depth'] = (self.receiver_depths,
                                    coordinate_unit('receiver_depth'))
        coords['phase_speed'] = (self.phase_speeds,
                                 coordinate_unit('phase_speed'))
        return coords

    def _export_attrs(self):
        attrs = super()._export_attrs()
        attrs['stabilizing_attenuation'] = self.stabilizing_attenuation
        attrs['title'] = self.title
        return attrs

    @classmethod
    def _from_export(cls, arrays, attrs):
        identity = cls._identity_from_attrs(attrs, cls._OWN_ATTRS)
        identity['source_depths'] = arrays['source_depth']
        return cls(data=arrays['data'], phase_speeds=arrays['phase_speed'],
                   receiver_depths=arrays['receiver_depth'],
                   times=arrays.get('time'),
                   stabilizing_attenuation=float(
                       attrs.get('stabilizing_attenuation', 0.0)),
                   title=str(attrs.get('title', '')), **identity)

    def to_dict(self) -> dict:
        """Serialise to plain arrays: ``data``, ``phase_speeds``,
        ``receiver_depths``, ``times``, ``stabilizing_attenuation``,
        ``title`` and the identity, as :meth:`Field.to_dict` writes it.
        ``np.savez(f, **d)`` stores it; read it back with
        ``np.load(f, allow_pickle=True)`` into :meth:`from_dict`."""
        return {
            'data': self.data.copy(),
            'phase_speeds': self.phase_speeds.copy(),
            'receiver_depths': self.receiver_depths.copy(),
            'times': None if self.times is None else self.times.copy(),
            'stabilizing_attenuation': self.stabilizing_attenuation,
            'title': self.title,
            **self._identity_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GreensFunction":
        """Reconstruct a :class:`GreensFunction` from :meth:`to_dict` output,
        or from the mapping ``np.load(f, allow_pickle=True)`` returns for a
        file written with ``np.savez(f, **gf.to_dict())``. A file written
        before the payload was named ``data`` holds it as ``values``.

        Parameters
        ----------
        d : mapping
            :meth:`to_dict` output, or the mapping ``np.load`` returns for it.
        """
        d = cls._unwrap_saved(
            d, payload=('data', 'values', 'phase_speeds', 'receiver_depths'))
        return cls(data=d['data'] if 'data' in d else d['values'],
                   phase_speeds=d['phase_speeds'],
                   receiver_depths=d['receiver_depths'], times=d['times'],
                   stabilizing_attenuation=d['stabilizing_attenuation'],
                   title=d['title'], **cls._identity_from_dict(d))
