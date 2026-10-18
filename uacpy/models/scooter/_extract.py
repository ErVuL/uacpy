"""The Hankel transform of Scooter's Green's function onto the receiver
ranges, and the phase-speed pass band of the kernel taper applied before it."""

import numpy as np

from uacpy.core.exceptions import ConfigurationError, ModelExecutionError
from uacpy.core.results._superposition import _stack_fields


def taper_bounds(phase_speeds, *, taper):
    """``(c_low, c_high)`` phase-speed bounds for the kernel taper.

    COA Sect. 4.5 prescribes the cure: since the wavenumber beyond which
    the integrand is negligible "will depend on range, and for multiple
    ranges it is not desirable to truncate at different wavenumbers", the
    kernel is "forced to gradually vanish" at one fixed edge. The roll-off
    is Hanning (:func:`~uacpy.core.acoustics.wavenumber_taper`, mirroring
    ``fieldsco.m:taper``); a rectangular edge's sidelobes fall at
    6 dB/octave against a Hann edge's 18 (Abraham, Sect. 4.10), i.e.
    ``1/x`` against ``1/x^3``.

    ``taper`` is a fraction of the wavenumber span applied at EACH edge.
    The fraction is taken in ``k`` while the bounds are returned as phase
    speeds, so ``omega`` cancels: they depend only on the deck's own
    ``c_low`` / ``c_high`` and are identical at every frequency of a
    broadband sweep. ``taper=0`` disables it and restores the rectangular
    cut.

    COA also warns that the choice of the truncation point itself "is not
    easily automated" -- tapering smooths an edge, it does not recover
    spectrum that was never computed. Widen ``c_high`` when the field is
    built close to a boundary, where steep and evanescent components still
    carry energy at the edge.
    """
    if taper <= 0.0:
        return None, None
    c = np.asarray(phase_speeds, dtype=float)
    if c.size and not np.all(np.isfinite(c) & (c > 0.0)):
        # ``wavenumber_taper`` indexes the RAW grid, so filtering here and
        # letting it index the unfiltered one turns a bad .grn into a
        # numpy ValueError or ZeroDivisionError from inside the transform.
        raise ConfigurationError(
            f"Scooter(taper={taper:.4g}): the Green's function's "
            f"phase-speed grid holds non-finite or non-positive values, "
            f"so the taper's edges cannot be located. Re-run the solver, "
            f"or pass taper=0 to transform it untapered.",
            remediation="A .grn with a corrupt cVec usually means the "
                        "run was interrupted; delete it and re-run.",
        )
    if c.size >= 4:
        inv_lo, inv_hi = 1.0 / c.max(), 1.0 / c.min()   # k/omega at each edge
        span = inv_hi - inv_lo
    if c.size < 4 or span <= 0.0:
        why = (f"holds only {c.size} value(s), too few to place a roll-off "
               f"at each edge" if c.size < 4 else
               "spans a single speed, so there is no edge to roll off")
        raise ConfigurationError(
            f"Scooter(taper={taper:.4g}): the Green's function's "
            f"phase-speed grid {why}, so the taper cannot be applied.",
            remediation=("Pass taper=0 to transform this Green's function "
                         "untapered, or give the deck more wavenumbers (a "
                         "wider c_low/c_high window or a longer RMax), so a "
                         "roll-off fits at each edge."),
        )
    return (1.0 / (inv_hi - taper * span),
            1.0 / (inv_lo + taper * span))


def assemble_field_from_grn(greens_function, source, receiver, broadband_mode,
                            *, taper, spectrum, model_name, log):
    """Hankel-transform the Green's function onto the receiver ranges.

    Broadband transforms every frequency in the ``.grn`` at once; the
    narrowband path transforms the single frequency slice. ``taper`` and
    ``spectrum`` are the model's knobs; ``log`` is the
    model's logger.
    """
    c_low, c_high = taper_bounds(greens_function.phase_speeds, taper=taper)
    if c_low is not None:
        log(f"Kernel taper: Hanning roll-off over "
            f"{taper:.4g} of the wavenumber span at each edge "
            f"(pass band {c_low:.1f}-{c_high:.1f} m/s)")
    transform_kwargs = dict(
        source_type=source.source_type,
        spectrum=spectrum,
        c_low=c_low, c_high=c_high,
    )
    if broadband_mode:
        log(f"Transforming {greens_function.n_frequencies} frequencies to "
            f"range domain...")
        return greens_function.to_transfer_function(
            receiver.ranges, **transform_kwargs)
    log("Transforming to range domain (direct-DFT Hankel transform)...")
    nsd = len(greens_function.source_depths)
    # One slab per source depth of the deck, on the Source's own depth
    # axis (the .grn stores the depths in float32).
    depths = np.atleast_1d(np.asarray(source.depths, dtype=float))
    if depths.size != nsd:
        raise ModelExecutionError(
            model_name, return_code=0, stdout=None,
            stderr=(f"the Green's function holds {nsd} source depths for "
                    f"a Source of {depths.size}."))
    fields = [greens_function.to_field(receiver.ranges, source_depth_idx=i,
                                       **transform_kwargs)
              for i in range(nsd)]
    if nsd == 1:
        return fields[0]
    return _stack_fields(fields, depths, coordinate_name='source_depth',
                         who=f"{model_name}.run")
