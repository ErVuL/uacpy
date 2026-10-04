"""Read an Acoustics-Toolbox ``.env`` deck back into uacpy carriers.

:func:`read_env` is the inverse of the KRAKEN-family and Bellhop deck
writers, and reads a deck any other tool wrote the way the Acoustics-Toolbox
programs read it: it returns the
:class:`~uacpy.core.environment.Environment`,
:class:`~uacpy.core.source.Source` and :class:`~uacpy.core.receiver.Receiver`
the deck describes, plus the solver options it carries. A construct those
carriers cannot hold exactly raises
:class:`~uacpy.core.exceptions.UnsupportedFeatureError` naming it, rather
than being approximated — with one exception: a Francois-Garrison ``'F'``
deck reads as :class:`~uacpy.core.absorption.FrancoisGarrison` with the
deck's water, evaluated at each depth. The deck's ``z_bar`` (the one depth
the solver evaluates the formula at for the whole column) is dropped with a
``FallbackWarning``, unless the deck is the broadband one uacpy writes for
that law (``z_bar`` at mid-water column).

The record order follows AT's own reader (``misc/ReadEnvironmentMod.f90``
for the KRAKEN family, ``Bellhop/ReadEnvironmentBell.f90`` for Bellhop):
title, frequency, NMedia, TopOpt and its follow-up rows, one mesh line plus
SSP rows per medium, BotOpt and the bottom half-space row, then the
program's own tail — ``cLow cHigh``, ``RMax``, source and receiver depths
(and the broadband frequency vector) for KRAKEN/Scooter; source depths,
receiver depths and ranges, RunType and the beam block for Bellhop.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, FileFormatError, IOWarning,
    UnsupportedFeatureError,
)
from uacpy.core.units import km_to_m
from uacpy.io.bellhop_writer import BELLHOP_SSP_GUARD_RANGE_FACTOR
from uacpy.core.deck_limits import AT_DB_PER_NEPER
from uacpy.io.at_codes import (
    BOUNDARY_TYPES_BY_CODE, SSP_INTERP_BY_CODE, AttenuationUnits,
)
from uacpy.io._fortran_helpers import (
    PARSE_ERRORS, _read_vector_values, expand_repeat_counts, fortran_float,
    list_directed_int, read_list_directed_values, read_vector,
    split_fortran_tokens, strip_fortran_comment, strip_fortran_quotes,
    typed_format_error,
)

#: Boundary letter (TopOpt(2) / BotOpt(1)) → ``acoustic_type``, from the one
#: letter table :data:`uacpy.io.at_codes.BOUNDARY_TYPES_BY_CODE`. Bellhop's
#: ``'G'`` (grain size) reads as the half-space its formulas give; see
#: :func:`_grain_size_row`.
_BOUNDARY_CODE_TO_TYPE = {
    code: kind.value for code, kind in BOUNDARY_TYPES_BY_CODE.items()}

#: The values AT's row variables ``alphaR betaR rhoR alphaI betaI`` hold
#: before the deck's first row: ``cp cs rho alphaP alphaS``
#: (``misc/ReadEnvironmentMod.f90:44-48``; ``Bellhop/sspMod.f90:21``).
_ROW_SEED = (1500.0, 0.0, 1.0, 0.0, 0.0)


#: TopOpt(3) attenuation units whose ``CRCI`` conversion equals a dB/wavelength
#: value at each row's own wave speed, whatever the frequency
#: (:attr:`AttenuationUnits.converts_to_db_per_wavelength`); see
#: :func:`_db_per_wavelength`.
_UNITS_AS_DB_PER_WAVELENGTH = tuple(
    unit.value for unit in AttenuationUnits
    if unit.converts_to_db_per_wavelength)


def _db_per_wavelength(unit: str, alpha: float, c: float) -> float:
    """``alpha`` in TopOpt(3)'s ``unit``, as dB/wavelength at wave speed ``c``.

    ``CRCI`` (``misc/AttenMod.f90``) turns every unit into Np/m. For
    ``'W'`` that is ``α·f/(8.6858896·c)``; ``'F'`` (dB/(m·kHz)) gives
    ``α·f/8685.8896``, ``'Q'`` gives ``ω/(2·c·Q)`` and ``'L'`` (loss
    parameter) gives ``α·ω/c``. Setting each equal to the ``'W'`` form
    leaves ``α·c/1000``, ``8.6858896·π/Q`` and ``8.6858896·2π·α``: no
    frequency, so one dB/wavelength value reproduces the deck at every
    frequency. ``CRCI`` then scales by ``c²``, so a zero wave speed (a fluid's
    shear column) carries no attenuation in any unit, and a zero ``Q`` is
    skipped (``c * alpha /= 0``).
    """
    if unit == 'W':
        return alpha
    if c == 0.0:
        return 0.0
    if unit == 'F':
        return alpha * c / 1000.0
    if unit == 'Q':
        return AT_DB_PER_NEPER * np.pi / alpha if alpha != 0.0 else 0.0
    return AT_DB_PER_NEPER * 2.0 * np.pi * alpha


class _Rows:
    """The ``z cp cs rho alphaP alphaS`` rows of one deck, read in order.

    AT READs every SSP row and both half-space rows into the same variables
    ``alphaR betaR rhoR alphaI betaI`` (``misc/sspMod.f90:334``,
    ``misc/ReadEnvironmentMod.f90:285``, ``Bellhop/sspMod.f90:903``,
    ``Bellhop/ReadEnvironmentBell.f90:474``), seeded once per deck with
    :data:`_ROW_SEED`. A ``/`` leaves every field it does not reach at the
    value of the previous READ: the previous row of the medium, the last
    row of the medium above, or the top half-space. ``carry`` holds those
    values in the deck's own attenuation unit; each returned row has its
    attenuations in dB/wavelength.
    """

    def __init__(self, fid, path, unit: str):
        self.fid, self.path, self.unit = fid, path, unit
        self.carry = list(_ROW_SEED)

    def read(self, what: str, z_fill: float) -> Tuple[float, ...]:
        vals = _values(_record(self.fid, what, self.path))
        raw = [z_fill, *self.carry]
        raw[:len(vals)] = vals[:6]
        self.carry = raw[1:]
        z, cp, cs, rho, ap, as_ = raw
        return (z, cp, cs, rho, _db_per_wavelength(self.unit, ap, cp),
                _db_per_wavelength(self.unit, as_, cs))


def _record(fid, what: str, path) -> str:
    """The next non-blank record; a list-directed numeric READ skips blank
    records, so a blank line between blocks is not a record."""
    while True:
        line = fid.readline()
        if line == '':
            raise FileFormatError(
                f"read_env: {path} ends before the {what} record.",
                remediation="The deck is truncated or is not an "
                            "Acoustics-Toolbox .env.")
        if line.strip():
            return line


def _skip_blank(fid) -> bool:
    """Move to the start of the next non-blank record; ``False`` when only
    blank records remain."""
    while True:
        pos = fid.tell()
        line = fid.readline()
        if line == '':
            return False
        if line.strip():
            fid.seek(pos)
            return True


def _vector(fid, what: str, path) -> np.ndarray:
    """One AT ``ReadVector``: a count record, then the values
    (:func:`~uacpy.io._fortran_helpers.read_vector`), with the file and the
    vector named when either is missing or malformed."""
    if not _skip_blank(fid):
        raise FileFormatError(
            f"read_env: {path} ends before the {what} count record.",
            remediation="The deck is truncated or is not an "
                        "Acoustics-Toolbox .env.")
    try:
        values, _ = read_vector(fid)
    except FileFormatError as exc:
        raise FileFormatError(
            f"read_env: {path}, {what}: {exc.message}",
            remediation=exc.remediation) from exc
    return np.asarray(values, dtype=float)


def _values(line: str) -> List[float]:
    """The numeric values of one list-directed record, up to its ``/``."""
    body = strip_fortran_comment(line).split('/', 1)[0]
    tokens = body.replace(',', ' ').split()
    return [fortran_float(t) for t in expand_repeat_counts(tokens)]


def _option(line: str, width: int) -> str:
    """A quoted option string, blank-padded to ``width`` positions."""
    return strip_fortran_quotes(line).ljust(width)


def _option_and_values(line: str, width: int) -> Tuple[str, List[float]]:
    """A record holding an option then numbers (``'A~' 0.5``,
    ``'MS' 1.0 100.0``): the option blank-padded to ``width``, and the
    values up to a ``/``. The trailing ``! …`` comment is dropped first,
    so a quote inside it is not read as part of the record."""
    tokens = split_fortran_tokens(strip_fortran_comment(line))
    option = strip_fortran_quotes(tokens[0]).ljust(width) if tokens else ''
    values = []
    for tok in expand_repeat_counts(tokens[1:]):
        if tok.startswith('/'):
            break
        values.append(fortran_float(tok.split('/', 1)[0]))
        if '/' in tok:
            break
    return option.ljust(width), values


def _halfspace(row, roughness: float = 0.0, grain_size_phi=None):
    from uacpy.core.boundary import BoundaryProperties
    _z, cp, cs, rho, ap, as_ = row
    return BoundaryProperties(
        acoustic_type='half-space', sound_speed=cp, shear_speed=cs,
        density=rho, attenuation=ap, shear_attenuation=as_,
        roughness=roughness, grain_size_phi=grain_size_phi)


def _grain_size_row(fid, path, what: str):
    """Bellhop's ``'G'`` boundary: a ``z Mz`` record whose half-space is the
    UW-APL grain-size formulas ``Bellhop/ReadEnvironmentBell.f90:494-530``
    evaluate. The sound speed is the ratio times the 1500 m/s that code
    uses (not the water's), the density is the ratio as written, and the
    loss parameter it hands ``CRCI`` with unit ``'L'`` becomes
    dB/wavelength by :func:`_db_per_wavelength`. Returns the half-space row
    and ``Mz``."""
    z, mz = _values(_record(fid, f"{what} grain-size", path))[:2]
    if -1 <= mz < 1:
        vr = 0.002709 * mz ** 2 - 0.056452 * mz + 1.2778
        rhor = 0.007797 * mz ** 2 - 0.17057 * mz + 2.3139
    elif 1 <= mz < 5.3:
        vr = (-0.0014881 * mz ** 3 + 0.0213937 * mz ** 2
              - 0.1382798 * mz + 1.3425)
        rhor = (-0.0165406 * mz ** 3 + 0.2290201 * mz ** 2
                - 1.1069031 * mz + 3.0455)
    else:
        vr = -0.0024324 * mz + 1.0019
        rhor = -0.0012973 * mz + 1.1565
    if -1 <= mz < 0:
        alpha2_f = 0.4556
    elif 0 <= mz < 2.6:
        alpha2_f = 0.4556 + 0.0245 * mz
    elif 2.6 <= mz < 4.5:
        alpha2_f = 0.1978 + 0.1245 * mz
    elif 4.5 <= mz < 6.0:
        alpha2_f = 8.0399 - 2.5228 * mz + 0.20098 * mz ** 2
    elif 6.0 <= mz < 9.5:
        alpha2_f = 0.9431 - 0.2041 * mz + 0.0117 * mz ** 2
    else:
        alpha2_f = 0.0601
    cp = vr * 1500.0
    loss = alpha2_f * (vr / 1000) * 1500.0 * np.log(10.0) / (40.0 * np.pi)
    row = (z, cp, 0.0, rhor, _db_per_wavelength('L', loss, cp), 0.0)
    return row, mz


def _boundary_row(code: str, rows: _Rows, what: str):
    """The record a TopOpt(2) / BotOpt(1) letter makes ``TopBot`` read:
    ``(row, Mz)`` for ``'A'`` (``Mz`` None) and ``'G'``, else None.
    Unknown letters are refused here, before any record is consumed."""
    if code not in _BOUNDARY_CODE_TO_TYPE:
        raise UnsupportedFeatureError(
            'read_env', f"{what} boundary code {code!r}",
            alternatives=[f"one of {sorted(_BOUNDARY_CODE_TO_TYPE)}"],
            alternatives_label='codes')
    if code == 'A':
        # TopBot zeroes its depth before the READ
        # (ReadEnvironmentMod.f90:284); the other five fields carry over.
        return rows.read(f"{what} half-space", 0.0), None
    if code == 'G':
        return _grain_size_row(rows.fid, rows.path, what)
    return None


def _boundary(code: str, path: Path, row, roughness: float, suffix: str):
    """The boundary a TopOpt(2) / BotOpt(1) letter names: the half-space of
    its :func:`_boundary_row` for ``'A'``/``'G'``, a pointer at the sibling
    table for ``'F'``/``'P'``."""
    from uacpy.core.boundary import BoundaryProperties
    if row is not None:
        values, mz = row
        return _halfspace(values, roughness, grain_size_phi=mz)
    if code in ('F', 'P'):
        table = path.with_suffix('.irc' if code == 'P' else suffix)
        return BoundaryProperties(
            acoustic_type=_BOUNDARY_CODE_TO_TYPE[code],
            reflection_file=str(table), roughness=roughness)
    return BoundaryProperties(acoustic_type=_BOUNDARY_CODE_TO_TYPE[code],
                              roughness=roughness)


def _volume_absorption(code: str, fid, path):
    """TopOpt(4) and the follow-up rows ``ReadTopOpt`` consumes: the law,
    and the ``z_bar`` of a ``'F'`` row (``None`` otherwise)."""
    from uacpy.core.absorption import Biological, FrancoisGarrison, Thorp
    if code == ' ':
        return None, None
    if code == 'T':
        return Thorp(), None
    if code == 'F':
        t, s, ph, z_bar = _values(_record(fid, 'Francois-Garrison', path))[:4]
        return FrancoisGarrison(temperature=t, salinity=s, pH=ph), z_bar
    if code == 'B':
        n_bio = list_directed_int(_record(fid, 'biological count', path))
        return Biological(layers=[
            tuple(_values(_record(fid, 'biological layer', path))[:5])
            for _ in range(n_bio)]), None
    raise UnsupportedFeatureError(
        'read_env', f"volume-attenuation code TopOpt(4) = {code!r}",
        alternatives=["' ' (none)", "'T' (Thorp)", "'F' (Francois-Garrison)",
                      "'B' (biological)"],
        alternatives_label='codes')


def _warn_if_z_bar_is_dropped(path, absorption, z_bar: float,
                              water_depth: float, broadband: bool) -> None:
    """A ``'F'`` deck reads as the Francois-Garrison law with its water,
    evaluated at each depth. That is the deck again only when uacpy writes
    the same one back: a broadband deck (``TopOpt(6)='B'``) whose ``z_bar``
    is mid-water column (``oalib_writer.francois_garrison_deck_depth``).
    Otherwise the deck's ``z_bar`` is dropped, and said so."""
    if broadband and abs(z_bar - 0.5 * water_depth) <= _Z_BAR_PRINT_TOL_M:
        return
    warnings.warn(
        f"read_env: {path.name} carries Francois-Garrison absorption "
        f"('F': T={absorption.temperature:g} °C, "
        f"S={absorption.salinity:g} psu, pH={absorption.pH:g}, "
        f"z_bar={z_bar:g} m), which the solver evaluates at z_bar and applies "
        f"at every depth (misc/AttenMod.f90:148-160). The FrancoisGarrison "
        f"law returned evaluates the formula at each depth with this water; "
        f"z_bar is dropped, so a run of it absorbs differently away from "
        f"{z_bar:g} m.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)


#: The ``z_bar`` resolution of a ``'F'`` row as ``write_fg_params`` prints it
#: (``.4f``), within which it is the deck's mid-water column.
_Z_BAR_PRINT_TOL_M = 1e-4


def _medium(rows: _Rows, index: int):
    """One medium: its mesh line and SSP rows down to the medium's base.

    Returns ``(n_mesh, sigma, base_depth, rows)``, ``rows`` an ``(n, 6)``
    array of ``z cp cs rho alphaP alphaS`` (attenuations in dB/wavelength).
    """
    fid, path = rows.fid, rows.path
    head = _values(_record(fid, f"medium {index} mesh", path))
    if len(head) < 3:
        raise FileFormatError(
            f"read_env: {path} medium {index} mesh line carries "
            f"{len(head)} values; it is 'NMesh sigma depth'.")
    n_mesh, sigma, base = int(head[0]), float(head[1]), float(head[2])
    medium = []
    z = 0.0
    while True:
        row = rows.read(f"medium {index} SSP row", z)
        medium.append(row)
        z = row[0]
        # ReadEnvironmentMod stops the medium on the row that reaches its
        # base depth; compare at the deck's 1e-6 m print resolution.
        if z >= base - 1e-6 * max(1.0, abs(base)):
            return n_mesh, sigma, base, np.asarray(medium, dtype=float)


def _sediment_layer(rows: np.ndarray, top: float, base: float, sigma: float,
                    index: int, path):
    from uacpy.core.boundary import SedimentLayer
    props = rows[:, 1:]
    if not np.allclose(props, props[0], rtol=1e-9, atol=1e-12):
        raise UnsupportedFeatureError(
            'read_env', f"{path}: sediment medium {index} varies with depth "
            f"(a gradient layer); a SedimentLayer is uniform",
            alternatives=["split the medium into uniform media in the deck"],
            alternatives_label='decks')
    cp, cs, rho, ap, as_ = props[0]
    return SedimentLayer(thickness=base - top, sound_speed=cp, density=rho,
                         attenuation=ap, shear_speed=cs,
                         shear_attenuation=as_, roughness=sigma)


def _nodes_for_switches(switches: np.ndarray, first: float,
                        gap: float) -> Optional[np.ndarray]:
    """Column ranges whose nearest-column midpoints fall on ``switches``.

    A long-format ``.bty`` holds each row's half-space over the segment to
    its right (``bellhop.f90:478``, ``GetBotSeg``), so its properties step
    at the row ranges. A :class:`~uacpy.core.bottom.Bottom` switches midway
    between its column ranges ``p``. Both steps agree when
    ``(p[j-1] + p[j]) / 2 == switches[j-1]``, which fixes every ``p`` from
    ``p[0]``: ``p[j] = (-1)**j p[0] + c[j]``. Each ``p[j] - p[j-1] >= gap``
    bounds ``p[0]`` from one side; ``first`` is taken when it lies inside
    the bounds, else their centre. None when no ``p[0]`` satisfies them.
    """
    n = switches.size + 1
    sign = np.ones(n)
    offset = np.zeros(n)
    for j in range(1, n):
        sign[j] = -sign[j - 1]
        offset[j] = 2.0 * switches[j - 1] - offset[j - 1]
    lo, hi = 0.0, np.inf
    for j in range(1, n):
        a = sign[j] - sign[j - 1]
        b = gap - (offset[j] - offset[j - 1])
        if a > 0:
            lo = max(lo, b / a)
        else:
            hi = min(hi, b / a)
    if lo > hi:
        return None
    if lo <= first <= hi:
        p0 = first
    else:
        p0 = 0.5 * (lo + hi) if np.isfinite(hi) else lo
    return sign * p0 + offset


def _range_dependent_halfspace(rows: np.ndarray, unit: str, roughness: float,
                               path):
    """The bottom a long-format ``.bty`` describes under an ``'A'`` BotOpt.

    ``rows`` is ``(7, N)``: range (m), depth, ``cp cs rho alphaP alphaS``
    in ``bdryMod.f90:200-201`` order, attenuations in TopOpt(3)'s unit
    (``bellhop.f90:192-199`` passes them through ``CRCI``). Consecutive
    equal rows are one step. One step is a range-independent half-space;
    more become a :class:`~uacpy.core.bottom.Bottom` whose columns sit
    where :func:`_nodes_for_switches` puts them.
    """
    from uacpy.core.bottom import Bottom
    from uacpy.core.deck_limits import DECK_RANGE_RESOLUTION_M
    props = np.array([
        (cp, cs, rho, _db_per_wavelength(unit, ap, cp),
         _db_per_wavelength(unit, as_, cs))
        for cp, cs, rho, ap, as_ in rows[2:].T])
    steps = [0] + [i for i in range(1, props.shape[0])
                   if not np.array_equal(props[i], props[i - 1])]
    values = props[steps]
    if len(steps) == 1:
        cp, cs, rho, ap, as_ = values[0]
        return _halfspace((0.0, cp, cs, rho, ap, as_), roughness)
    ranges = _nodes_for_switches(rows[0, steps[1:]], float(rows[0, 0]),
                                 1.5 * DECK_RANGE_RESOLUTION_M)
    if ranges is None:
        raise UnsupportedFeatureError(
            'read_env', f"{path.with_suffix('.bty')}: long-format half-space "
            f"steps at {rows[0, steps[1:]]} m that no set of column ranges "
            f"reproduces (a Bottom switches midway between its columns)",
            alternatives=["a short-format .bty with one half-space"],
            alternatives_label='files')
    return Bottom.from_halfspaces(
        ranges, sound_speed=values[:, 0], shear_speed=values[:, 1],
        density=values[:, 2], attenuation=values[:, 3],
        shear_attenuation=values[:, 4], roughness=roughness)


def _without_guard_rows(depths: np.ndarray, speeds: np.ndarray):
    """Drop the leading SSP rows above ``z = 0`` that repeat the row below.

    ``write_bellhop_env_file`` prepends such a row to raise the top of the
    profile over the altimetry's crests (``bdryMod.f90:113-114`` aborts
    otherwise), and writes it again from the Environment. It repeats the
    next row's speed, so it carries nothing the Environment does not;
    a row above the surface with its own speed is kept.
    """
    while (depths.size > 2 and depths[0] < 0.0
           and np.array_equal(speeds[0], speeds[1])):
        depths, speeds = depths[1:], speeds[1:]
    return depths, speeds


def _without_guard_columns(ranges: np.ndarray, speeds: np.ndarray,
                           r_box: Optional[float]):
    """Drop the ``.ssp`` columns ``write_bellhop_env_file`` adds to bracket
    Bellhop's ray box.

    Two guards, each a copy of its neighbour: a column at a negative range
    (``-1.1·r_box``, so a back-scattered ray stays inside Quad's box), and a
    last column at ``1.1·r_box`` when the profiles stop short of the box.
    A negative-range column that differs from the first profile at or past
    range 0 is data, and is left for the carrier to refuse.
    """
    while (ranges.size > 1 and ranges[0] < 0.0
           and np.array_equal(speeds[:, 0], speeds[:, 1])):
        ranges, speeds = ranges[1:], speeds[:, 1:]
    if (r_box is not None and ranges.size > 2
            and np.isclose(ranges[-1], BELLHOP_SSP_GUARD_RANGE_FACTOR * r_box,
                           rtol=1e-6, atol=0.0)
            and np.array_equal(speeds[:, -1], speeds[:, -2])):
        ranges, speeds = ranges[:-1], speeds[:, :-1]
    return ranges, speeds


def _sibling_flp(flp: Path):
    """The receivers and source options of a sibling ``.flp``.

    Two programs read a ``.flp`` beside a KRAKEN-family ``.env``, in two
    layouts. KRAKEN's ``field.exe`` (``KrakenField/field.f90:64-152``):
    title, option, ``MLimit``, profiles, then the receiver grid; parsed by
    :func:`~uacpy.io.oalib_reader.read_flp`. Scooter's ``fields.exe``
    (``Scooter/fields.f90:61-97``): option, then the receiver ranges in km;
    the depths are the ``.env``'s. The second record tells them apart: a
    quoted option for ``field.exe``, a range count for ``fields.exe``.

    Returns ``(option, depths, ranges)``, ``depths`` None for the Scooter
    layout; ``option[0] == 'X'`` is a line source in both, and a ``'*'`` in
    column 3 (``field.exe``) or 4 (``fields.exe``) the source beam pattern.
    """
    with open(flp, 'r', encoding='utf-8', errors='replace') as fid:
        first = _record(fid, 'first', flp)
        second = _record(fid, 'second', flp)
        if second.lstrip()[:1] in ("'", '"'):
            from uacpy.io.oalib_reader import _parse_flp
            data = _parse_flp(flp)
            opt = data['opt'].ljust(4)
            pos = data['pos']['r']
            return (opt[0], opt[2] == '*',
                    np.asarray(pos['z'], dtype=float),
                    np.asarray(pos['r'], dtype=float))
        fid.seek(0)
        _record(fid, 'option', flp)
        ranges_km = _vector(fid, 'receiver ranges', flp)
    opt = strip_fortran_quotes(first).ljust(4)
    return opt[0], opt[3] == '*', None, km_to_m(ranges_km)


def _kraken_tail(fid, path, topopt: str) -> Dict[str, Any]:
    """KRAKEN / Scooter / SPARC after the bottom: ``cLow cHigh``, ``RMax``,
    source and receiver depths, and the frequency vector of a broadband
    (TopOpt(6) ``'B'``) deck."""
    c_low, c_high = _values(_record(fid, 'cLow cHigh', path))[:2]
    rmax_km = _values(_record(fid, 'RMax', path))[0]
    if not _skip_blank(fid):
        raise UnsupportedFeatureError(
            'read_env', f"{path}: the deck ends after RMax, with no source "
            f"or receiver block; that is a BOUNCE deck, whose output is a "
            f"reflection table",
            alternatives=["uacpy.io.read_reflection_coefficient on the "
                          ".brc BOUNCE writes"],
            alternatives_label='readers')
    tail: Dict[str, Any] = {
        'options': {'c_low': c_low, 'c_high': c_high,
                    'rmax_m': float(km_to_m(rmax_km))},
        'source_depths': _vector(fid, 'source depths', path),
        'receiver_depths': _vector(fid, 'receiver depths', path),
        'receiver_ranges': None,
    }
    if topopt[5] == 'B':
        tail['frequencies'] = _vector(fid, 'frequencies', path)
    return tail


def _bellhop_tail(fid, path) -> Dict[str, Any]:
    """Bellhop after the bottom (``ReadEnvironmentBell.f90:108-240``):
    source depths, receiver depths and ranges, RunType, the launch angles,
    ``STEP ZBOX RBOX``, and the Cerveny beam records."""
    tail: Dict[str, Any] = {
        'source_depths': _vector(fid, 'source depths', path),
        'receiver_depths': _vector(fid, 'receiver depths', path),
        'receiver_ranges': km_to_m(_vector(fid, 'receiver ranges', path)),
    }
    run = _option(_record(fid, 'RunType', path), 7)
    # ReadRunType (ReadEnvironmentBell.f90:361-377) stops on any other
    # RunType(1); a record that is not one is not a 2-D Bellhop tail.
    if run[0] not in 'REISCAa':
        raise FileFormatError(
            f"read_env: {path} has {run!r} where the Bellhop RunType "
            f"record belongs; RunType(1) is one of R E I S C A a.")
    n_beams = list_directed_int(_record(fid, 'NBeams', path))
    # angleMod.f90:54-58 READs alpha(1:MAX(3, Nalpha)) list-directed: the
    # values may wrap across records and a '/' ends them early.
    angles, _ = _read_vector_values(fid, max(3, n_beams))
    if angles.size == 0:
        raise FileFormatError(
            f"read_env: {path} gives no beam angles after NBeams.")
    # One READ takes all three (ReadEnvironmentBell.f90:146), wrapping
    # across records: AT's decks write them on one line, uacpy's on two.
    step, z_box, r_box_km = read_list_directed_values(
        fid, 3, 'STEP ZBOX RBOX', path)
    options = dict(
        run_type=run[0], beam_type=run[1], source_type=run[3],
        grid_type=run[4], source_beam_pattern=run[2] == '*',
        beam_shift=run[6] == 'S', n_beams=n_beams,
        launch_angles=(float(angles[0]), float(angles[-1])), ray_step=float(step),
        z_box=float(z_box), r_box=float(km_to_m(r_box_km)))
    # ReadEnvironmentBell.f90:192-221 reads the two Cerveny records for a
    # 'C'/'R' beam unless the run is a ray trace; uacpy's writer emits them
    # for a ray trace too, so there they are read when present.
    if run[1].upper() in ('C', 'R') and (run[0] != 'R' or _skip_blank(fid)):
        shape, shape_values = _option_and_values(
            _record(fid, 'beam shape', path), 2)
        eps, r_loop_km = shape_values[:2]
        tokens = split_fortran_tokens(strip_fortran_comment(
            _record(fid, 'Nimage iBeamWindow Component', path)))
        options.update(
            beam_width_type=shape[0], beam_curvature=shape[1],
            eps_multiplier=eps, r_loop=float(km_to_m(r_loop_km)),
            n_image=int(fortran_float(tokens[0])),
            ib_win=int(fortran_float(tokens[1])),
            component=(strip_fortran_quotes(tokens[2])
                       if len(tokens) > 2 else 'P'))
    tail['options'] = options
    tail['run'] = run
    return tail


def _is_bellhop3d_tail(fid, path) -> bool:
    """Whether the tail is BELLHOP3D's: source x and y, source and receiver
    depths, receiver ranges and bearings (``ReadSxSy``, ``ReadSzRz``,
    ``ReadRcvrRanges``, ``ReadRcvrBearings``), then a RunType whose sixth
    letter is ``'3'`` (3-D) or ``'2'`` (Nx2D)."""
    try:
        for what in ('source x', 'source y', 'source depths',
                     'receiver depths', 'receiver ranges',
                     'receiver bearings'):
            _vector(fid, what, path)
        run = _option(_record(fid, 'RunType', path), 7)
    except (FileFormatError, *PARSE_ERRORS):
        return False
    return run[5] in ('2', '3')


@typed_format_error
def read_env(
    filepath: Union[str, Path],
    model: Optional[str] = None,
):
    """Read an Acoustics-Toolbox ``.env`` into uacpy carriers.

    Reads a KRAKEN-family (Kraken, Scooter, SPARC) or 2-D Bellhop ``.env``
    as those programs do (see the module docstring for the record order),
    and the sibling files the deck names: a Bellhop ``.bty``/``.ati``
    (``'~'``/``'*'`` flags; a long-format ``.bty`` under an ``'A'`` bottom
    is a range-dependent seabed), a ``.ssp`` for a ``'Q'`` range-dependent
    SSP, and a ``.brc``/``.trc``/``.irc`` reflection table (referenced by
    path, not read). A row a ``/`` ends early keeps the previous READ's
    values, across media and into both half-spaces. Attenuation in
    TopOpt(3) ``'F'``, ``'Q'`` or ``'L'`` is converted to the
    dB/wavelength it equals at each row's own wave speed; Bellhop's
    ``'G'`` grain-size boundary becomes the half-space its formulas give.

    A sibling ``.flp`` supplies the receivers of a KRAKEN-family deck. For
    ``field.exe``'s layout the receivers come from the ``.flp`` and the
    ``.env``'s receiver depths — the mode-tabulation grid — go to
    ``options['mode_depths']``; for Scooter's ``fields.exe`` layout the
    ``.flp`` gives the ranges. Its option makes a line source (``'X'``) or
    points ``beam_pattern`` at the ``.sbp`` (``'*'``). Without a readable
    one the ``.env`` depths are the receivers and no ranges are known.

    uacpy's own Bellhop decks read back as the Environment they were
    written from: the ``.ssp`` guard columns (a copy of the first profile
    at ``-1.1·r_box``, of the last at ``1.1·r_box``) and the guard SSP row
    above the altimetry crests are dropped.

    Parameters
    ----------
    filepath : str or Path
        The ``.env`` file.
    model : {'kraken', 'bellhop'}, optional
        Which program's tail to read after the bottom block. ``None`` tells
        them apart by the first record after it: KRAKEN's ``cLow cHigh``
        holds two values, Bellhop's source-depth count one.

    Returns
    -------
    env : Environment
        Water SSP (range-dependent from a ``.ssp``), water density from the
        first SSP row, bathymetry (flat at the deck's water depth, or the
        ``.bty``), altimetry (the ``.ati``, turned to the public positive-up
        convention), sediment layers, bottom and surface boundaries,
        volume absorption (TopOpt(4)), or a
        :class:`~uacpy.core.absorption.ConstantAbsorption` when the water's
        ``alphaP`` column is a non-zero constant, or the column as a table
        at the deck frequency (a measured
        :class:`~uacpy.core.absorption.AbsorptionCoefficient`, no model,
        in dB/wavelength) when it varies with depth, and
        ``name`` = the title.
    source : Source
        Source depths and frequency (the broadband vector for a ``'B'``
        deck); Bellhop's RunType(4) ``'X'`` makes it a line source and
        RunType(3) ``'*'`` points ``beam_pattern`` at the sibling ``.sbp``.
    receiver : Receiver
        Receiver depths, and ranges from the deck (Bellhop) or the ``.flp``.
    options : dict
        ``'model'``, ``'title'``, and the solver settings under the keyword
        names the matching writer takes: ``interp_ssp``, ``n_mesh`` (one per
        medium), and ``c_low``/``c_high``/``rmax_m`` (KRAKEN) or
        ``run_type``/``beam_type``/``source_type``/``grid_type``/
        ``n_beams``/``alpha``/``step``/``z_box``/``r_box`` (Bellhop, metres),
        plus the Cerveny/Gaussian beam-shape entries when the deck has them,
        and ``mode_depths`` for a KRAKEN deck read beside its ``.flp``.

    Raises
    ------
    FileFormatError
        A truncated or malformed deck; the error names the file and the
        record.
    UnsupportedFeatureError
        A construct an Environment cannot hold: attenuation in Np/m or dB/m
        (TopOpt(3) ``'N'``, ``'M'``, ``'m'``, another frequency law), the
        analytic (``'A'``) or 3-D hexahedral (``'H'``) SSP, a BELLHOP3D
        deck, a gradient sediment medium, a depth-varying water density, a
        depth-varying water attenuation on repeated depths, a long-format ``.ati`` under an ``'A'`` top, a BOUNCE
        deck (no source/receiver block), an unknown boundary or option
        code.
    ConfigurationError
        No file at ``filepath``, or values an Environment, Source or
        Receiver refuses (e.g. receivers above the surface, a water density
        outside 0.9-1.1 g/cm³).

    Warns
    -----
    FallbackWarning
        A Francois-Garrison (``'F'``) deck other than the broadband one uacpy
        writes (``z_bar`` mid-water column): its ``z_bar`` is dropped, and
        the law returned evaluates the formula at each depth.
    """
    from uacpy.core.absorption import AbsorptionCoefficient, ConstantAbsorption
    from uacpy.core.bottom import SeabedColumn
    from uacpy.core.environment import Environment, SoundSpeedProfile
    from uacpy.core.receiver import Receiver
    from uacpy.core.source import Source

    path = Path(filepath)
    if not path.is_file():
        raise ConfigurationError(
            f"read_env: no file at {str(path)!r}.",
            remediation="Check the path; a deck is read exactly as named.")
    if model is not None and model not in ('kraken', 'bellhop'):
        raise ConfigurationError(
            f"read_env: model must be 'kraken', 'bellhop' or None; got "
            f"{model!r}.")

    with open(path, 'r') as fid:
        title = strip_fortran_quotes(_record(fid, 'title', path))
        freq = fortran_float(_values(_record(fid, 'frequency', path))[0])
        n_media = list_directed_int(_record(fid, 'NMedia', path))
        topopt = _option(_record(fid, 'TopOpt', path), 7)
        ssp_code, top_code, unit_code, vol_code = topopt[:4]
        if ssp_code == 'H':
            raise UnsupportedFeatureError(
                'read_env', f"{path}: SSP interpolation TopOpt(1) = 'H', the "
                f"BELLHOP3D hexahedral (x, y, z) SSP; an Environment is 2-D",
                alternatives=["uacpy.io.read_ssp_3d for the deck's .ssp"],
                alternatives_label='readers')
        if ssp_code == 'A':
            raise UnsupportedFeatureError(
                'read_env', f"{path}: SSP TopOpt(1) = 'A', the analytic "
                f"profile compiled into the engine (sspMod's 'ANALYTIC'); "
                f"the deck holds no sound speeds",
                alternatives=sorted(SSP_INTERP_BY_CODE),
                alternatives_label='codes')
        if ssp_code not in SSP_INTERP_BY_CODE:
            raise UnsupportedFeatureError(
                'read_env', f"SSP interpolation TopOpt(1) = {ssp_code!r}",
                alternatives=sorted(SSP_INTERP_BY_CODE),
                alternatives_label='codes')
        if unit_code not in _UNITS_AS_DB_PER_WAVELENGTH:
            raise UnsupportedFeatureError(
                'read_env', f"attenuation unit TopOpt(3) = {unit_code!r}; "
                f"uacpy carries dB/wavelength, and this unit's attenuation "
                f"has another frequency law ('N' Np/m and 'M' dB/m are "
                f"constant in frequency, 'm' a power law)",
                alternatives=["'W' (dB/wavelength)", "'F' (dB/(m kHz))",
                              "'Q' (quality factor)", "'L' (loss parameter)"],
                alternatives_label='units')
        absorption, z_bar = _volume_absorption(vol_code, fid, path)
        rows = _Rows(fid, path, unit_code)

        # The top half-space row sits here, between ReadTopOpt and the
        # medium loop (ReadEnvironmentMod.f90:285); its roughness is the
        # water medium's sigma, read below.
        top_row = _boundary_row(top_code, rows, 'top')

        n_mesh, depths_top = [], 0.0
        water = None
        layers = []
        surface_sigma = 0.0
        for m in range(1, n_media + 1):
            mesh, sigma, base, medium = _medium(rows, m)
            n_mesh.append(mesh)
            if m == 1:
                water, surface_sigma, depths_top = medium, sigma, base
                if z_bar is not None:
                    _warn_if_z_bar_is_dropped(path, absorption, z_bar, base,
                                              topopt[5] == 'B')
            else:
                layers.append(_sediment_layer(medium, depths_top, base,
                                              sigma, m, path))
                depths_top = base

        # BotOpt and the bottom roughness share one record: "'A~' 0.5".
        botopt, bot_sigma_vals = _option_and_values(
            _record(fid, 'BotOpt', path), 2)
        bot_sigma = bot_sigma_vals[0] if bot_sigma_vals else 0.0
        bottom_hs = _boundary(botopt[0], path,
                              _boundary_row(botopt[0], rows, 'bottom'),
                              bot_sigma, '.brc')

        # The program's own tail. KRAKEN's first record is cLow cHigh (two
        # values), Bellhop's the source-depth count (one).
        pos = fid.tell()
        if model is None:
            probe = _values(_record(fid, 'the record after the bottom', path))
            fid.seek(pos)
            model = 'kraken' if len(probe) >= 2 else 'bellhop'
        try:
            tail = (_kraken_tail(fid, path, topopt) if model == 'kraken'
                    else _bellhop_tail(fid, path))
        except (FileFormatError, *PARSE_ERRORS):
            fid.seek(pos)
            if _is_bellhop3d_tail(fid, path):
                raise UnsupportedFeatureError(
                    'read_env', f"{path}: a BELLHOP3D deck (source x-y, "
                    f"receiver bearings, RunType(6) '3' or Nx2D '2'); an "
                    f"Environment is 2-D",
                    alternatives=["uacpy.io.read_boundary_3d for its .bty",
                                  "uacpy.io.read_ssp_3d for its .ssp"],
                    alternatives_label='readers') from None
            raise

    if model == 'kraken' and 'G' in (top_code, botopt[0]):
        raise UnsupportedFeatureError(
            'read_env', f"{path}: boundary code 'G' (grain size) in a KRAKEN "
            f"deck; only Bellhop reads it (TopBot in misc/ReadEnvironmentMod"
            f".f90 refuses it)",
            alternatives=sorted(set(_BOUNDARY_CODE_TO_TYPE) - {'G'}),
            alternatives_label='codes')
    options: Dict[str, Any] = {
        'model': model, 'title': title,
        'interp_ssp': SSP_INTERP_BY_CODE[ssp_code], 'n_mesh': n_mesh,
        **tail['options'],
    }
    frequencies: Union[float, np.ndarray] = tail.get('frequencies', freq)
    source_depths = tail['source_depths']
    receiver_depths = tail['receiver_depths']
    receiver_ranges = tail['receiver_ranges']
    source_kwargs: Dict[str, Any] = {}
    if model == 'kraken':
        flp = path.with_suffix('.flp')
        if flp.is_file():
            # field.exe takes the receivers from the .flp; the .env's
            # receiver depths are then the grid the modes were tabulated
            # on, kept as options['mode_depths']. Scooter's fields.exe
            # takes only the ranges from it.
            # A .flp neither program can read (e.g. the pre-ReadVector
            # 'RMIN RMAX NR' record) leaves the deck's own receivers.
            try:
                coords, beam_pattern, flp_depths, flp_ranges = \
                    _sibling_flp(flp)
            except (FileFormatError, *PARSE_ERRORS) as exc:
                warnings.warn(
                    f"read_env: the sibling {flp} is neither a field.exe nor "
                    f"a fields.exe .flp ({getattr(exc, 'message', exc)}); "
                    f"the receivers are the "
                    f"deck's depths, with no ranges.",
                    IOWarning, skip_file_prefixes=USER_FRAME_SKIP)
                coords, beam_pattern, flp_depths = 'R', False, None
            else:
                receiver_ranges = flp_ranges
            if flp_depths is not None:
                options['mode_depths'] = receiver_depths
                receiver_depths = flp_depths
            if coords == 'X':
                source_kwargs['source_type'] = 'line'
            if beam_pattern:
                source_kwargs['beam_pattern'] = str(path.with_suffix('.sbp'))
    else:
        run = tail['run']
        if run[3] == 'X':
            source_kwargs['source_type'] = 'line'
        if run[2] == '*':
            source_kwargs['beam_pattern'] = str(path.with_suffix('.sbp'))

    # --- the Environment ---------------------------------------------------
    water_depths, cp = water[:, 0], water[:, 1]
    rho = water[:, 3]
    if not np.allclose(rho, rho[0]):
        raise UnsupportedFeatureError(
            'read_env', f"{path}: the water density varies with depth "
            f"({rho.min():g} to {rho.max():g} g/cm³); an Environment carries "
            f"one water density",
            alternatives=["a constant density column"],
            alternatives_label='decks')
    alpha_water = water[:, 4]
    if absorption is None and np.any(alpha_water != 0.0):
        if np.allclose(alpha_water, alpha_water[0]):
            absorption = ConstantAbsorption(
                value_dB_per_wavelength=float(alpha_water[0]))
        elif np.all(np.diff(water_depths) > 0):
            # The column as the deck states it: dB/wavelength per row at the
            # deck frequency, which the Environment takes as a table.
            absorption = AbsorptionCoefficient(
                frequencies=np.array([float(freq)]),
                data=np.asarray(alpha_water, dtype=float)[:, None],
                units='dB/wavelength',
                depths=np.asarray(water_depths, dtype=float))
        else:
            raise UnsupportedFeatureError(
                'read_env', f"{path}: a depth-varying water attenuation "
                f"column on repeated depths", alternatives=[
                    "TopOpt(4) volume absorption",
                    "an attenuation column on increasing depths"],
                alternatives_label='decks')

    if ssp_code == 'Q':
        from uacpy.io.oalib_reader import _parse_ssp_2d
        ssp2d = _parse_ssp_2d(path.with_suffix('.ssp'))
        ssp_ranges, speeds = _without_guard_columns(
            np.asarray(ssp2d['r_prof'], dtype=float),
            np.asarray(ssp2d['c_mat'], dtype=float),
            options.get('r_box'))
        ssp_depths, speeds = _without_guard_rows(water_depths, speeds)
        ssp = SoundSpeedProfile(depths=ssp_depths, sound_speed=speeds,
                                ranges=ssp_ranges)
    else:
        ssp_depths, speeds = _without_guard_rows(water_depths, cp)
        ssp = SoundSpeedProfile(depths=ssp_depths, sound_speed=speeds)

    bathymetry: Any = float(water[-1, 0])
    altimetry = None
    # BotOpt(2) and TopOpt(5) are Bellhop's boundary-file flags; the
    # KRAKEN family reads BotOpt(1:8) and acts on BotOpt(1) alone
    # (ReadEnvironmentMod.f90:124-128).
    # A long-format file's geoacoustic columns replace the half-space
    # segment by segment (bellhop.f90:192-199, :474-481); only an 'A'
    # boundary reflects off them.
    if model == 'bellhop' and botopt[1] in ('~', '*'):
        from uacpy.io.bathy_io import _parse_bathymetry
        bty, _ = _parse_bathymetry(path.with_suffix('.bty'))
        bathymetry = bty[:2, 1:-1].T
        if bty.shape[0] > 2 and botopt[0] == 'A':
            bottom_hs = _range_dependent_halfspace(bty[:, 1:-1], unit_code,
                                                   bot_sigma, path)
    if model == 'bellhop' and topopt[4] in ('~', '*'):
        from uacpy.io.bathy_io import _parse_altimetry
        ati, _ = _parse_altimetry(path.with_suffix('.ati'))
        if ati.shape[0] > 2 and top_code == 'A':
            raise UnsupportedFeatureError(
                'read_env', f"{path.with_suffix('.ati')}: a long-format "
                f"altimetry file under an 'A' top, a range-dependent top "
                f"half-space; an Environment's surface is one boundary",
                alternatives=["a short-format (range, depth) .ati"],
                alternatives_label='files')
        # The .ati is positive-down on Bellhop's z axis; Environment
        # altimetry is height, positive up.
        altimetry = np.column_stack([ati[0, 1:-1], -ati[1, 1:-1]])

    surface = _boundary(top_code, path, top_row, surface_sigma, '.trc')
    bottom = (SeabedColumn(layers=layers, halfspace=bottom_hs) if layers
              else bottom_hs)
    env = Environment(
        bathymetry=bathymetry, ssp=ssp, altimetry=altimetry, bottom=bottom,
        surface=surface, absorption=absorption, name=title,
        water_density=float(rho[0]))
    source = Source(depths=np.asarray(source_depths, dtype=float),
                    frequencies=frequencies, **source_kwargs)
    receiver = Receiver(depths=np.asarray(receiver_depths, dtype=float),
                        ranges=receiver_ranges)
    return env, source, receiver, options
