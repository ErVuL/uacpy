"""The environments a Kraken deck cannot carry, refused by name, and the
reductions that make one carriable: the elastic stacks, krakenc over a hard
or free floor, roughness on a tabulated top, the
r = 0 profile of the MODES path and the elastic sub-bottom receivers."""

import warnings
import numpy as np
from typing import Optional
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._projection import _max_roughness, _smooth_surface
from uacpy.core.run_settings import Notice
from uacpy.core.run_settings import RunMode
from uacpy.core.bathymetry import Bathymetry
from uacpy.core.environment import Environment
from uacpy.core.receiver import Receiver
from uacpy.core.results import Field
from uacpy.io.at_codes import boundary_code
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, UnsupportedFeatureError,
    ValidityWarning,
)
from uacpy.io.oalib_writer import deck_depth, writable_layers


#: ``acoustic_type`` values that put a tabulated reflection coefficient on a
#: boundary — AT letters ``'F'`` (``.brc`` / ``.trc``) and ``'P'`` (``.irc``).
#: Neither is usable on real ``kraken.exe``: ``Kraken/kraken.f90:47-48`` stops
#: outright on a bottom ``'F'`` or a top ``'P'``, and the two mirror cases pass
#: that guard only to be thrown away — every mode-finding call passes
#: ``ComplexFlag = .FALSE.`` (``Kraken/kraken.f90:447,451,613,654,785-786,
#: 793-794``) and ``Kraken/BCImpedanceMod.f90:113-116,121-125`` then replaces
#: the tabulated impedance with a rigid boundary (``f = 0, g = 1``).
#: ``krakenc.exe`` honours both (``Kraken/BCImpedancecMod.f90:88-106``).
_REFLECTION_TABLE_TYPES = ('file', 'precalc')

#: Top ``TopOpt(2:2)`` letters whose branch of ``Kraken/kraken.f90:850-867``
#: (and ``krakenc.f90:848-865``) leaves ``rho1``/``eta1Sq`` non-zero, so the
#: Kuperman-Ingenito determinant ``Del = rho1*eta2 + rho2*eta1``
#: (``Kraken/Scattering.f90:21``) is non-zero and the sea-surface roughness in
#: ``SSP%sigma(1)`` reaches the eigenvalue perturbation. Every other letter
#: falls to ``CASE DEFAULT``, which zeroes both, so ``KupIng`` returns its
#: initialised ``0.0D0`` (``Scattering.f90:17,23``) — see
#: :func:`drop_roughness_on_tabulated_top`.
_ROUGHNESS_BEARING_TOP_CODES = ('A', 'V', 'R')


def has_elastic_surface(env, *, collapse) -> bool:
    """Whether the surface the writer will emit carries shear.

    Reduced the way the deck sees it: Kraken carries a single global top,
    so a range-dependent surface is first collapsed by the configured
    ``collapse['surface']`` method.
    """
    surface = env.surface
    if surface.is_range_dependent:
        surface = surface.collapse_range(collapse['surface'])
    return bool(surface.is_elastic)


def deck_bottom_columns(env: Environment, run_mode) -> list:
    """The seabed columns the deck(s) for ``run_mode`` will carry.

    The two paths differ, and both are right for what they build.
    :func:`modes_single_profile` samples the r = 0 profile of every
    range-dependent quantity — SSP, bathymetry, surface — so its bottom is
    the r = 0 column too; taking any other column there would pair a
    seabed from one range with a water column from another. Every field
    path segments a range-dependent bottom instead: each profile block of
    the multi-profile ``.env`` is a full environment read by its own
    ``ReadEnvironment`` call (``kraken.f90:42-46``), and
    :func:`segment_environment_by_range` hands each one
    ``env.bottom.at(range=r)``. So a field run carries every column.

    Guards that predict what the deck will carry have to ask here rather
    than assume one of the two.
    """
    bottom = env.bottom
    if run_mode == RunMode.MODES or not bottom.is_range_dependent:
        return [bottom.at(range=0.0)]
    return list(bottom.columns)


def reject_acoustic_below_elastic(env: Environment,
                                   run_mode=None, *, model_name) -> None:
    """Reject any acoustic medium sitting below an elastic one.

    Two failure modes, one shape — a solid-over-liquid interface inside
    the bottom stack:

    * An elastic layer (shear_speed > 0) terminated by a **fluid
      halfspace**: krakenc.exe spins forever in setup, so the caller
      would wait out the whole ``timeout`` for nothing.
    * An elastic layer with a **fluid layer under it**: krakenc.exe
      aborts (SIGABRT, "double free or corruption"). ``kraken.f90:170``
      sets ``LastAcoustic`` to the *deepest* acoustic medium, so
      ``FirstAcoustic .. LastAcoustic`` spans the elastic medium in
      between and ``Vector``'s loops (``:560-568``, ``:632-651``) walk it
      as if it were acoustic.

    An elastic layer over a rigid or vacuum floor is refused too, as
    what it is: krakenc runs on it but the field is 0.8-1.7 dB (median)
    off Scooter.

    Elastic half-spaces and elastic-over-elastic stacks are fine, as is a
    fluid layer *above* an elastic one.

    ``run_mode`` selects the columns (see :func:`deck_bottom_columns`)
    so the columns tested are the ones the deck will carry: r = 0 alone
    for MODES, every column for a segmented field run.
    """
    # A bottom with no layered column anywhere cannot carry the
    # solid-over-liquid stack this guard rejects.
    if not env.bottom.is_layered:
        return
    for col in deck_bottom_columns(env, run_mode):
        reject_solid_over_liquid_column(col, model_name=model_name)


def reject_solid_over_liquid_column(col, *, model_name) -> None:
    """The per-column test behind :func:`reject_acoustic_below_elastic`."""
    shear = [layer.shear_speed > 0
             for layer in col.layers]
    if not any(shear):
        return
    first_elastic = shear.index(True)
    fluid_below = not all(shear[first_elastic:])
    floor = str(col.halfspace.acoustic_type)
    halfspace_fluid = (col.halfspace.shear_speed == 0
                       and floor not in ('rigid', 'vacuum'))
    if fluid_below:
        raise UnsupportedFeatureError(
            model_name,
            "a fluid sediment layer below an elastic one "
            "(solid-over-liquid interface inside the bottom stack) — "
            "kraken.f90:170 spans the elastic medium with "
            "FirstAcoustic..LastAcoustic and krakenc.exe aborts on it",
            alternatives=['Scooter', 'OAST'],
        )
    if halfspace_fluid:
        raise UnsupportedFeatureError(
            model_name,
            "an elastic sediment layer over a fluid halfspace "
            "(solid-over-liquid bottom interface) — krakenc.exe does not "
            "converge on it",
            alternatives=['Scooter', 'OAST'],
        )
    if floor in ('rigid', 'vacuum'):
        # Measured with this refusal lifted: a 20 m layer (cp 1800,
        # cs 400) under 100 m of water at 50 and 200 Hz ran, 0.8-1.7 dB
        # median and 4-6 dB at the 90th percentile off Scooter, where
        # the same layer over an elastic half-space agrees with it.
        raise UnsupportedFeatureError(
            model_name,
            f"an elastic sediment layer over a {floor} floor — "
            f"krakenc.exe runs on it but returns a field 0.8-1.7 dB "
            f"(median) and 4-6 dB (90th percentile) off Scooter",
            alternatives=['Scooter', 'OAST'],
        )


def reject_rough_elastic_layer(env: Environment,
                                run_mode=None, *, model_name) -> None:
    """Reject roughness on the interface above an elastic sediment layer.

    ``kraken.f90:178`` / ``krakenc.f90:182`` stop with *"Rough elastic
    interfaces are not allowed"* whenever an ELASTIC medium carries a
    non-zero ``SSP%sigma``, and ``sigma`` belongs to the interface at the
    **top** of its own medium — which the writer takes from that layer's
    own ``roughness`` (``oalib_writer:1072``). The model's spec advertises
    ``rough_bottom`` and ``elastic_media`` separately and both are true;
    it is only this pairing that the binaries refuse.

    A rough elastic *half-space* is fine: its ``sigma(NMedia+1)`` sits on
    the ``BotOpt`` line and feeds the Kirchhoff (``KupIng``) correction
    instead of the medium loop.
    """
    if not env.bottom.is_layered:
        return
    layers = [(i, layer)
              for col in deck_bottom_columns(env, run_mode)
              for i, layer in enumerate(col.layers)]
    for i, layer in layers:
        shear = float(layer.shear_speed)
        sigma = float(layer.roughness)
        if shear > 0.0 and sigma != 0.0:
            raise UnsupportedFeatureError(
                model_name,
                f"roughness ({sigma:g} m) on the interface above elastic "
                f"sediment layer {i + 1} (shear_speed={shear:g} m/s) — "
                f"kraken.f90:178 / krakenc.f90:182 stop with 'Rough "
                f"elastic interfaces are not allowed'",
                alternatives=[
                    'Set that layer roughness to 0 (a rough elastic '
                    'half-space is accepted)',
                    'Scooter', 'OAST',
                ],
            )


def drop_roughness_on_tabulated_top(env, *, model_name):
    """``env`` with the sea-surface roughness zeroed, and a warning, when
    the top boundary condition cannot carry it.

    ``write_ssp_section`` in ``oalib_writer.py`` writes
    ``env.surface.roughness`` as ``SSP%sigma(1)`` on the water mesh line
    for every ``TopOpt`` letter, and ``Kraken/kraken.f90:902`` feeds that
    slot into ``KupIng``. Whether it reaches the answer is decided one
    branch earlier: ``kraken.f90:850-867`` selects on ``HSTop%BC``, and the
    ``CASE DEFAULT`` a tabulated top ``'F'`` lands in sets
    ``rho1 = eta1Sq = 0``. ``Kraken/Scattering.f90:21`` then forms
    ``Del = rho1*eta2 + rho2*eta1``, which is exactly zero
    (``ScatterRoot(0) = 0``), the ``IF ( Del /= 0.0D0 )`` at
    ``Scattering.f90:23`` is false, and ``KupIng`` returns the ``0.0D0`` it
    was initialised to at ``Scattering.f90:17``. ``krakenc.f90:848-865,899``
    is the same shape. Measured on a 100 m Pekeris guide at 100 Hz with a
    19-angle ``.trc`` top: ``roughness=0.5`` against ``0.0`` moves the field
    by exactly 0.0 under a tabulated top and by 3.3 % under a vacuum one.

    Dropping it here rather than declaring ``rough_surface`` conditionally
    keys the decision on the resolved environment, which is what decides
    the ``TopOpt`` letter — one ``Kraken`` instance can be run against a
    tabulated-top env and a vacuum-top one, and only the first loses the
    roughness.
    """
    sigma = _max_roughness(env.surface.nodes)
    if not sigma:
        return env
    acoustic_type = env.surface.acoustic_type
    code = boundary_code(acoustic_type)
    if code in _ROUGHNESS_BEARING_TOP_CODES:
        return env
    env.surface = _smooth_surface(env.surface)
    warnings.warn(
        f"{model_name} cannot apply sea-surface roughness to a "
        f"{acoustic_type!r} top boundary: kraken.f90:864-866 zeroes the "
        f"density and vertical wavenumber a tabulated top has none of, so "
        f"the Kuperman-Ingenito scatter term (Scattering.f90:23) is "
        f"identically zero. env.surface.roughness={sigma:g} m was dropped "
        f"rather than written into a deck that discards it. Use a vacuum, "
        f"rigid or half-space surface to keep it.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )
    return env


def reflection_table_boundaries(env, *, top_reflection_file) -> list:
    """Boundaries of ``env`` that carry a tabulated reflection coefficient.

    Returns human-readable labels (empty when there are none). Every range
    node is inspected, matching the ``any``-semantics of
    ``env.has_elastic_*``: the dispatch runs before the environment is
    projected, so a table anywhere on the axis still ends up in the deck.
    """
    labels = []
    if top_reflection_file is not None:
        labels.append("surface (Kraken(top_reflection_file=...) → '.trc')")
    for p in env.surface.nodes:
        if p.acoustic_type in _REFLECTION_TABLE_TYPES:
            labels.append(f"surface (acoustic_type={p.acoustic_type!r})")
    for column in env.bottom.columns:
        if column.halfspace.acoustic_type in _REFLECTION_TABLE_TYPES:
            labels.append(
                f"bottom (acoustic_type={column.halfspace.acoustic_type!r})")
    return sorted(set(labels))


def modes_single_profile(env: Environment, *, collapse,
                         model_name) -> Environment:
    """Normal modes are range-independent; reduce any range-dependent env
    to its r=0 profile (a run()-time numerical requirement — the field
    path segments RD natively, the modes solve cannot).

    r = 0 is applied to every quantity here, overriding the configured
    ``collapse`` methods rather than consulting them (see also
    :func:`deck_bottom_columns`): the source sits in one real column, and
    taking its SSP from one reduction while the bathymetry, bottom and
    surface came from another would assemble a waveguide that exists at no
    range at all. The warning names whichever configured method the
    sampling dropped, so a setting that was asked for and not applied says
    so instead of passing silently.
    """
    if not env.is_range_dependent:
        return env
    overridden = [
        f"collapse[{key!r}]={collapse[key]!r}"
        for key, is_rd in (
            ('ssp', env.ssp.is_range_dependent),
            ('bathymetry', env.bathymetry.varies_with_range),
            ('bottom_range',
             env.bottom is not None and env.bottom.is_range_dependent),
            ('surface', env.surface.is_range_dependent),
        )
        if is_rd and collapse[key] != 'r0'
    ]
    dropped = (f" This drops {', '.join(overridden)}: the modes solve "
               f"applies no collapse method, taking every quantity from "
               f"the source's own column." if overridden else '')
    warnings.warn(
        f"{model_name}: normal modes are range-independent; sampling "
        f"the r=0 profile of the range-dependent environment.{dropped}",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )
    ssp = env.ssp.collapse_range('r0') if env.ssp.is_range_dependent else env.ssp
    bottom = env.bottom
    if bottom is not None and bottom.is_range_dependent:
        bottom = bottom.collapse_range('r0')
    # Carry the full context of the original env: altimetry passes
    # through untouched so ``_project_environment`` still sees it and
    # emits its collapse disclosure, and the geolocation / date /
    # provenance fields survive onto the reduced env (the r=0 seafloor
    # keeps its bathymetry's records; the reduced ssp, bottom and surface
    # keep their own).
    reduced = Environment(
        name=env.name,
        bathymetry=Bathymetry(
            ranges=np.array([0.0]),
            depths=np.array([float(env.bathymetry.eval(range=0.0))]),
            data_sources=env.bathymetry.data_sources),
        ssp=ssp,
        altimetry=env.altimetry,
        bottom=bottom,
        surface=env.surface.collapse_range('r0'),
        absorption=env.absorption,
        water_density=env.water_density,
        location=env.location,
        transect=env.transect,
        date=env.date,
        extra_data_sources=env.extra_data_sources,
    )
    return reduced


def mask_elastic_mode_depths(modes, env, *, model_name) -> None:
    """NaN the eigenfunction samples that fall inside an elastic medium.

    The modes path sizes its depth grid on the whole media stack
    (``_grid.dense_mode_depths``), but the binary
    tabulates the eigenvector over the ACOUSTIC media only, and the
    samples below the last acoustic node are a straight-line
    extrapolation of the two above it — see
    :func:`elastic_depth_intervals`. The half-space record still names
    the sediment base as the domain bottom, so nothing downstream marks
    them. Mark them here, with the same no-data policy the field paths
    apply through :func:`partition_elastic_subbottom`.
    """
    phi = modes.phi
    z = np.atleast_1d(np.asarray(modes.depths, dtype=float))
    if z.size == 0 or not env.bottom.is_layered:
        return
    spans = elastic_depth_intervals(
        env, deck_bottom_columns(env, RunMode.MODES)[0])
    if not spans:
        return
    mask = np.zeros(z.shape, dtype=bool)
    for top, base in spans:
        mask |= (z > top) & (z <= base)
    if not mask.any():
        return
    phi = np.array(phi, dtype=complex)
    phi[mask, ...] = np.nan
    modes.phi = phi
    warnings.warn(
        f"{model_name}: {int(mask.sum())} mode-shape depth(s) lie in "
        "an elastic sub-bottom medium, which kraken/krakenc does not "
        "tabulate — the samples the binary returns there are a linear "
        "extrapolation of the acoustic column, not a mode shape. "
        "Returning NaN at those depths. Pass "
        "Kraken(mode_depths=...) to keep the grid in the acoustic "
        "media, or use Scooter / OAST for the elastic sub-bottom.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def elastic_depth_intervals(env, column):
    """Depth intervals ``(top, bottom]`` whose mode samples are not written.

    The deck's AT media are the water column, then each layer of ``column``
    at its written thickness (:func:`deck_depth`); the half-space below
    them is a boundary condition, not a medium.

    An elastic medium is not *mis*-tabulated in a kraken/krakenc ``.mod``;
    it is absent. ``kraken.f90:558-568`` builds the mode's depth vector over
    ``FirstAcoustic .. LastAcoustic`` only, and the file's ``Material``
    record (``:594``) is written over the same span — so
    ``ReadModes.f90:246-250`` never sees an ``'ELASTIC'`` entry, ``TufLuk``
    stays false and the stress-displacement compaction at ``:296-331``
    (a KRAKEL output format, which a kraken/krakenc ``.mod`` can never
    carry) is never reached. What the caller gets instead is
    ``calculateweights.f90``'s documented out-of-domain extrapolation:
    past the last acoustic node the index sticks at ``Nx-1`` and the
    weight runs above 1, so ``PhiTab`` (``kraken.f90:674``) continues the
    final two acoustic samples in a straight line. The acoustic media
    above the elastic one are unaffected, which is why the exclusion is
    per medium rather than "everything under the water column".

    Open at the top and closed at the bottom because an interface depth is
    tabulated twice, once per adjoining medium, and
    ``calculateweights.f90:43-49`` brackets it with ``w = 1`` on the upper
    copy: a receiver at an elastic medium's top interface reads the
    acoustic sample above it, one at its bottom interface reads the
    elastic sample.
    """
    spans = []
    top = deck_depth(float(env.depth))
    layers = writable_layers(column) if column is not None else []
    for i, layer in enumerate(layers):
        bottom = deck_depth(top + float(layer.thickness))
        if layer.shear_speed > 0.0:
            # A receiver below the deepest medium is extrapolated off
            # Phi(NTot) (``ReadModes.f90:279``), the last tabulated
            # sample — so an elastic bottom-most medium takes the
            # half-space under it with it.
            last = (i == len(layers) - 1)
            spans.append((top, float('inf') if last else bottom))
        top = bottom
    return spans


def partition_elastic_subbottom(env, receiver, elastic_spans, *, model_name):
    """Split receivers into depths field.exe can evaluate and depths inside
    an elastic medium, which it cannot. Returns ``(compute_receiver,
    keep_mask, notice)`` with ``keep_mask`` marking evaluable depths in
    the original ordering and ``notice`` the ``(note, warning)`` the run
    announces, or ``(receiver, None, None)`` when no split is needed.

    ``elastic_spans`` is the ``(top, bottom]`` interval list from
    :func:`elastic_depth_intervals`. ``True`` stands for the whole
    sub-bottom, for a caller that knows an elastic medium is present but
    not where it sits.
    """
    if elastic_spans is True:
        elastic_spans = [(float(env.depth), float('inf'))]
    if not elastic_spans:
        return receiver, None, None
    depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
    keep = np.ones(depths.shape, dtype=bool)
    for top, bottom in elastic_spans:
        keep &= ~((depths > top) & (depths <= bottom))
    if keep.all():
        return receiver, None, None
    notice = Notice(
        f"{int((~keep).sum())} receiver depth(s) in an elastic "
        f"sub-bottom medium come back NaN",
        f"{model_name}: {int((~keep).sum())} receiver depth(s) lie in "
        "an elastic sub-bottom medium, where field.exe cannot evaluate "
        "the field; returning NaN there. Depths in a fluid sediment layer "
        "*above* the elastic one are computed normally (a fluid layer "
        "below it is refused up front — see "
        "``reject_acoustic_below_elastic``).", FallbackWarning,
    )
    # ``misc/SourceReceiverPositions.f90:212`` ERROUTs on an empty receiver
    # vector, so a deck whose every requested depth sits in the sub-bottom
    # still needs one legal depth. Mid-water is arbitrary: with ``keep``
    # all-False ``reinsert_nan_depths`` discards the computed column and
    # returns NaN at every original depth.
    compute_depths = depths[keep] if keep.any() else np.array([0.5 * env.depth])
    compute_receiver = Receiver(
        depths=compute_depths, ranges=receiver.ranges,
    )
    return compute_receiver, keep, notice


def reinsert_nan_depths(field, receiver, keep):
    """Map ``field`` (computed on the water-column receivers) back onto the
    original receiver depth axis, NaN at the dropped sub-bottom depths.
    No-op when ``keep`` is None."""
    if keep is None:
        return field
    full_depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
    d = field.to_dict()
    data = np.asarray(d['data'])
    full = np.full((full_depths.size,) + data.shape[1:], np.nan, dtype=data.dtype)
    if keep.any():
        full[keep, ...] = data
    d['data'] = full
    d['coords'] = {**d['coords'], 'depth': full_depths}
    return Field.from_dict(d)


def reject_krakenc_over_a_hard_or_free_floor(env, run_mode, *, forced_backend,
                                             leaky_modes,
                                             top_reflection_file) -> None:
    """Refuse KRAKENC forced by a knob onto a rigid or vacuum seabed
    that ``kraken.exe`` answers.

    A rigid or vacuum floor has no half-space speed to bound the mode
    search. ``kraken.exe`` solves it with the unbounded window (0.05 dB
    median against :func:`uacpy.analytic.ideal_waveguide` at 50 and
    200 Hz, on either floor), but KRAKENC's complex root finder over the
    fluid column does not, at any window: with the unbounded one it keeps
    no mode at 50 Hz, returns an all-NaN field on the vacuum floor and a
    field 6.6 dB (median) off on the rigid one at 200 Hz; with
    ``c_high`` 3000, 1e4 or 1e5 m/s it runs and returns fields 0.7 to
    9 dB (median) off with no warning. So ``backend='krakenc'`` and
    ``leaky_modes=True`` (which forces KRAKENC, and asks for modes that
    leak into a half-space a rigid or vacuum floor does not have) are
    refused wherever the environment itself would run on ``kraken.exe``.
    An environment that needs KRAKENC anyway (an elastic medium or a
    reflection table) keeps it, with the finite window
    :func:`_window.phase_speed_window` gives it; ``leaky_modes=True`` there,
    under an elastic ice canopy, is refused too, since its unbounded
    window returned the finite window's field or no field
    (``log/fix-r2-models_scratch/k1_leaky_ice_rigid``,
    ``k2b_leaky_ice_vacuum``).
    """
    if forced_backend != 'krakenc' and not leaky_modes:
        return
    floors = sorted({
        str(column.halfspace.acoustic_type)
        for column in deck_bottom_columns(env, run_mode)
    } & {'rigid', 'vacuum'})
    if not floors:
        return
    if reflection_table_boundaries(env,
                                   top_reflection_file=top_reflection_file):
        return
    floor = ' / '.join(floors)
    if leaky_modes and env.surface.is_elastic:
        raise ConfigurationError(
            f"Kraken(leaky_modes=True) under an elastic ice canopy over "
            f"a {floor} seabed: the automatic krakenc route already "
            f"solves this environment with a finite window, and the "
            f"unbounded one leaky_modes asks for adds nothing to it — "
            f"where it finished it returned the same field (0.011-0.012 "
            f"dB median against Scooter, rigid and vacuum floors, 50-200 "
            f"Hz) in 8-10 s against under 0.1 s, and it did not finish "
            f"in 240 s (rigid, 50 Hz) or kept no mode (vacuum, 200 Hz).",
            remediation=(
                "Leave leaky_modes=False: the automatic route runs "
                "krakenc with the finite window here."),
        )
    if env.bottom.is_elastic or env.surface.is_elastic:
        return
    measured = (
        "KRAKENC over a fluid water column on a rigid or vacuum floor "
        "keeps no mode, returns an all-NaN field, or returns a field "
        "several dB off: against uacpy.analytic.ideal_waveguide it kept "
        "no mode at 50 Hz and was 6.6 dB (median) off on a rigid floor "
        "at 200 Hz, and no finite c_high repairs it (0.7-9 dB median off "
        "at 3000-1e5 m/s, with no warning)"
    )
    if leaky_modes:
        raise ConfigurationError(
            f"Kraken(leaky_modes=True) over a {floor} seabed: a rigid or "
            f"vacuum floor has no half-space for a mode to leak into, and "
            f"the krakenc solve leaky_modes forces fails there — "
            f"{measured}.",
            remediation=(
                "Leave leaky_modes=False: kraken.exe solves this seabed "
                "with the unbounded window (0.05 dB median against the "
                "analytic ideal waveguide)."),
        )
    raise ConfigurationError(
        f"Kraken(backend='krakenc') over a {floor} seabed: {measured}.",
        remediation=(
            "Use backend='kraken' or backend=None (the automatic dispatch "
            "picks kraken.exe here): it solves this seabed with the "
            "unbounded window, 0.05 dB median against the analytic ideal "
            "waveguide."),
    )


def backend_origin(env, *, forced_backend, leaky_modes,
                   top_reflection_file) -> str:
    """Why :meth:`select_backend` picked the binary it picked, for
    :attr:`KrakenSettings.backend_origin`."""
    if forced_backend is not None:
        return f"Kraken(backend={forced_backend!r})"
    if leaky_modes:
        return "leaky_modes=True: complex eigenvalues"
    if env.bottom.is_elastic or env.surface.is_elastic:
        return "an elastic medium: complex eigenvalues"
    tables = reflection_table_boundaries(
        env, top_reflection_file=top_reflection_file)
    if tables:
        return f"a reflection table on the {', '.join(tables)}"
    return "a fluid environment: real arithmetic"


def krakenc_incoherent_sum_notice(run_mode, backend: str,
                                   n_profiles: int, *, model_name
                                   ) -> Optional[Notice]:
    """``(note, warning)`` when field.exe's incoherent branch will square
    complex mode contributions without taking their magnitude first.

    ``field.f90:214-215`` picks the evaluator by profile count. The
    single-profile one, ``EvaluateMod.f90:66``, computes
    ``SQRT(SUM(z**2))`` — no ``ABS`` — which equals the energy sum
    ``SQRT(SUM(|z|**2))`` only for real mode functions, so krakenc's
    complex ``phi`` and ``k`` leave cross-mode phase inside the square.
    The multi-profile adiabatic evaluator, ``EvaluateADMod.f90:110``,
    uses ``SQRT(SUM(ABS(...)**2))`` and is sound on either backend; the
    multi-profile coupled one is unreachable here because
    ``field.f90:125-129`` refuses 'C' with 'I' and stage 2 rejects that
    pairing up front.
    """
    if run_mode != RunMode.INCOHERENT_TL or backend != 'krakenc':
        return None
    if int(n_profiles) > 1:
        return None
    return Notice(
        "INCOHERENT_TL on krakenc is not a strict energy sum on one "
        "profile",
        f"{model_name}: INCOHERENT_TL on the krakenc backend is "
        "not a strict incoherent sum for a range-independent run. "
        "field.exe sends the single-profile case to EvaluateMod.f90:66, "
        "whose Opt(4:4)='I' branch computes SQRT(SUM(z**2)) over the "
        "per-mode contributions — the energy sum SQRT(SUM(|z|**2)) only "
        "for real mode functions, and krakenc's phi and k are complex. "
        "Use backend='kraken' where the environment allows it, or "
        "RunMode.COHERENT_TL.", ValidityWarning,
    )
