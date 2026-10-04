"""The launch outputs of a Kraken run read into the package's conventions:
the ``.shd`` of a launch, its source and receiver, and the field assembled
from field.exe's pressure."""

import dataclasses
import numpy as np
from pathlib import Path
from uacpy.models._conventions import _line_source_unit_at_1m
from uacpy.models._extract import _restore_requested_axis
from uacpy.models._stacking import _slabs_of
from uacpy.core.run_settings import RunMode
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Field, PhaseReference
from uacpy.io.oalib_reader import read_shd_file, read_shd_bin
from uacpy.models.kraken._settings import KrakenLaunch, _BAND_ROUTES


#: Relative precision of a band read back from field.exe's ``.shd``: the deck
#: carries it as ``%.12g`` text (the frequency vector of
#: :mod:`uacpy.io.oalib_writer`), half a unit in the twelfth significant
#: digit, which AT reads into
#: REAL(KIND=8) (``misc/SourceReceiverPositions.f90:14, 65``) and writes back
#: unrounded (``misc/RWSHDFile.f90:106``).
_SHD_FREQUENCY_RTOL = 5e-12


def read_shd(shd_file: Path, launch: KrakenLaunch, *,
              one_bin: bool) -> dict:
    """The ``.shd`` field.exe wrote for ``launch``: every frequency's
    pressure of a band deck, one bin's pressure (``one_bin``: a band
    solved narrowband), or the TL reader's field (one slab per source
    depth)."""
    if launch.marched_frequencies is not None:
        freqs_read = np.asarray(read_shd_bin(str(shd_file)).frequencies,
                                dtype=float)
        # read_shd_bin returns pressure as (Ntheta, Nsz, Nrz, Nrr);
        # [0, 0] selects the single bearing and single source depth the
        # deck was written with. The band is labelled with the one the
        # deck was written from when the two agree to the deck's text.
        return {'frequencies': _restore_requested_axis(
                    freqs_read, np.asarray(launch.marched_frequencies, dtype=float),
                    _SHD_FREQUENCY_RTOL),
                'pressure': [read_shd_bin(str(shd_file), frequency=float(
                    fr)).pressure[0, 0, :, :] for fr in freqs_read]}
    if one_bin:
        return {'pressure': read_shd_bin(str(shd_file)).pressure[
            0, 0, :, :]}
    return {'shd': read_shd_file(shd_file)}


def launch_of(inputs) -> KrakenLaunch:
    """The resolved settings of the launch ``inputs`` are for."""
    return inputs.settings.engine.launches[inputs.launch]


def deck_source(inputs, launch: KrakenLaunch) -> Source:
    """The source the decks of ``launch`` are written for: the call's,
    or — for a band solved bin by bin, and a band's one propagating bin
    — the call's source at the bin's frequency."""
    if (inputs.settings.engine.route in _BAND_ROUTES
            and launch.marched_frequencies is None):
        return dataclasses.replace(
            inputs.source,
            frequencies=np.array([launch.deck_frequency]))
    return inputs.source


def evaluated_receiver(inputs) -> Receiver:
    """The receiver field.exe evaluates: the call's, or its depths
    outside every elastic sub-bottom medium
    (:attr:`KrakenSettings.evaluated_depths`)."""
    engine = inputs.settings.engine
    if engine.evaluated_depths is None:
        return inputs.receiver
    return Receiver(depths=np.array(engine.evaluated_depths),
                    ranges=inputs.receiver.ranges)


def assemble_field_from_shd(raw, source, receiver, is_rd,
                             n_profiles, run_mode,
                             line_c_source=None, source_rho=None, *,
                             backend, mode_coupling, result_kwargs,
                             stamp_result):
    """Build the result Field from field.exe's ``.shd`` as
    :meth:`_read_output` read it (``raw``): native-broadband
    ``(n_d, n_r, n_f)`` (``raw['frequencies']``), single-frequency
    complex pressure (``raw['pressure']``), or narrowband TL
    (``raw['shd']``). field.exe's modal sum differs
    from Scooter's Hankel path by an overall -1, negated here — times
    ``exp(-i*pi/4)`` for a line source — so every coherent branch carries
    the ``'travelling_wave'`` phase reference, upcast to complex128 (the
    ``.shd`` payload is complex64) so every uacpy engine returns one
    dtype. ``RunMode.INCOHERENT_TL`` instead yields real dB TL with no
    phase reference — a magnitude sum has no phase to reference.

    ``source_rho`` is ``ρ(z_s)`` (g/cm³) per source depth. field.exe
    leaves the ``1/ρ(z_s)`` of the modal sum out (``EvaluateMod.f90:34``
    has none, while the Kraken manual's TL formula and Jensen et al.
    eq. 5.14 carry it, and Scooter divides its forcing by ``rhoSz``), so
    its field is the unit-source field times ``ρ(z_s)``: 0.23 dB loud at
    the default 1.027 g/cm³ water, about 5 dB for a source in 1.8 g/cm³
    sediment. Every branch divides it out here.

    ``backend`` is the modes binary whose ``.mod`` field.exe summed
    (``'kraken'`` / ``'krakenc'``), the result's ``backend``."""
    # EvaluateMod.f90:34 applies one prefactor i*SQRT(2*pi)*EXP(i*pi/4)
    # to every source geometry; its pi/4 is the stationary-phase term of
    # the POINT-source Hankel asymptote. The 2-D line-source field has no
    # such term — its exact kernel already integrates to the modal sum —
    # so field.exe's 'X' branch comes out exp(+i*pi/4) ahead of Scooter's
    # line-source transform and needs the same exp(-i*pi/4) Bellhop
    # applies via _LINE_SOURCE_PHASE. 'point' / 'scaled' need only the
    # overall -1 shared by every branch.
    phase_corr = np.complex128(-1.0)
    if source.source_type == 'line':
        phase_corr = -np.exp(-1j * np.pi / 4.0)

    rho = np.atleast_1d(np.asarray(
        1.0 if source_rho is None else source_rho, dtype=float))

    # c(z_s) per source depth: k0 = 2πf/c(z_s) is taken at each slab's
    # own depth, as ``source_rho`` is.
    line_c = (None if line_c_source is None
              else np.atleast_1d(np.asarray(line_c_source, dtype=float)))

    def line_level(freqs, i_source=0):
        # field.exe's line field is Σ ψψ e^{ikx}/k_x (EvaluateMod.f90:36):
        # ×√k0 puts it at unit amplitude at 1 m, the package's convention
        # (_conventions._line_source_unit_at_1m); 1 for a point/scaled source.
        if line_c is None:
            return np.ones(np.atleast_1d(np.asarray(freqs, dtype=float)).size)
        return _line_source_unit_at_1m(
            float(line_c[i_source if line_c.size > 1 else 0]), freqs)
    if 'frequencies' in raw:
        freqs_read = raw['frequencies']
        # New layout: (n_d, n_r, n_f).
        p_stack = np.zeros(
            (len(receiver.depths), len(receiver.ranges), len(freqs_read)),
            dtype=np.complex128,
        )
        for i_freq, fr in enumerate(freqs_read):
            # field.exe already marches the engineering carrier
            # exp(-ikr) (EvaluateMod.f90:42); only its constant is off.
            # Its prefactor (EvaluateMod.f90:34) is i·√(2π)·e^{iπ/4}
            # = -√(2π)·e^{-iπ/4}, whereas the point-source modal sum
            # normalized to 1 m under this carrier carries
            # +√(2π)·e^{-iπ/4} in that slot (the e^{-iπ/4}/√(8π) of
            # the standard sum times the 4π the 1 m free-field
            # reference strips) — the form Scooter's Hankel path
            # returns. field.exe is therefore the wanted field times
            # -1 (times e^{iπ/4} for a line source), so ``phase_corr``
            # aligns the two; conjugating instead would flip the
            # already-correct carrier sign.
            p_stack[:, :, i_freq] = (phase_corr * line_level(fr)[0]
                                     / rho[0]
                                     * raw['pressure'][i_freq])
        field = Field(
            data=p_stack,
            coords={
                'depth': receiver.depths,
                'range': receiver.ranges,
                'frequency': freqs_read,
            },
            phase_reference=PhaseReference.TRAVELLING_WAVE,
            **result_kwargs(
                source,
                backend=backend,
                frequencies=freqs_read,
                mode_coupling=mode_coupling if is_rd else 'none',
                n_profiles=n_profiles,
                native_broadband=True,
            ),
        )
    elif 'pressure' in raw:
        # (nrz, nrr) at the deck's single bearing and source depth;
        # complex128 like every other engine.
        p = (phase_corr * line_level(np.atleast_1d(source.frequencies)[0])[0]
             / rho[0]
             * np.asarray(raw['pressure'], dtype=np.complex128))
        field = Field(
            data=p,
            coords={'depth': receiver.depths, 'range': receiver.ranges},
            **result_kwargs(
                source,
                backend=backend,
                frequencies=float(np.atleast_1d(source.frequencies)[0]),
                mode_coupling=mode_coupling if is_rd else 'none',
                n_profiles=n_profiles,
            ),
        )
        # Same negated-Hankel convention as the COHERENT_TL / broadband
        # branches, so the complex pressure carries one phase reference.
        field.phase_reference = PhaseReference.TRAVELLING_WAVE
    else:
        read = raw['shd']
        # One slab per source depth of the deck — read_shd_file stacks an
        # NSz > 1 .shd — each stamped with its own depth.
        slabs = _slabs_of(read)
        sources = ([source.at_depth(i) for i in range(len(slabs))]
                   if len(slabs) > 1 else [source])
        slab_rho = (rho if rho.size == len(slabs)
                    else np.full(len(slabs), rho[0]))
        for i_slab, (field, slab_source, rho_s) in enumerate(
                zip(slabs, sources, slab_rho)):
            # The line-source level (×√k0, see ``line_level``) is applied
            # ONCE here, before the INCOHERENT/COHERENT split, because both
            # branches below need it: COHERENT_TL keeps this payload as the
            # complex pressure and INCOHERENT_TL takes ``.dB`` of it. It is a
            # real, positive scalar, so it commutes with both the magnitude
            # sum and the ``phase_corr`` rotation, and it is exactly 1 for a
            # point/scaled source. Without it the narrowband line-source
            # result sat 10·log10(k0) dB away from this engine's own
            # broadband branch, from Scooter and from Bellhop — an offset
            # that changes with frequency (and changes sign at
            # k0 = 1, f = c(z_s)/2π), so it could not be read as a constant
            # convention difference.
            field.data = (line_level(np.atleast_1d(source.frequencies)[0],
                                     i_slab)[0]
                          / rho_s * np.asarray(field.data))
            if run_mode == RunMode.INCOHERENT_TL:
                # Opt(4:4)='I' returns SQRT(SUM(z**2)) over the per-mode
                # contributions with the range phase dropped
                # (EvaluateMod.f90:43,66); AT parks that in the complex .shd
                # slot, where its phase is an artefact. Store real dB TL so
                # the result claims only what it has.
                field.data = np.asarray(field.dB, dtype=float)
                # The unit was fixed at construction, from the complex
                # payload; the payload is its dB view from here on.
                field._unit = 'dB'
                phase_reference = None
            else:
                # field.exe emits the modal sum with a prefactor that differs
                # from Scooter's Hankel path by an overall -1 — times
                # e^{+iπ/4} for a line source (see the broadband branch
                # above). Apply ``phase_corr`` here too, upcast to
                # complex128, and tag travelling_wave so the COHERENT_TL
                # complex pressure carries the SAME phase convention and
                # dtype as the broadband / return_pressure branches and as
                # Scooter (|TL| is unchanged; this only fixes the complex
                # phase).
                field.data = phase_corr * np.asarray(
                    field.data, dtype=np.complex128)
                phase_reference = 'travelling_wave'
            stamp_result(
                field, slab_source, backend=backend,
                frequencies=float(np.atleast_1d(source.frequencies)[0]),
                phase_reference=phase_reference,
            )
            field.metadata['mode_coupling'] = (mode_coupling if is_rd
                                               else 'none')
            field.metadata['n_profiles'] = n_profiles
        field = read
    return field
