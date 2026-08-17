"""
Writer for the Collins-style ``ram.in`` text input shared by the RAM family
binaries uacpy dispatches to:

- ``ramgeo``      — fluid PE, flat surface, range-dependent layered bottom
- ``ramsurf1.5``  — fluid PE, *variable* surface (rough surface / beach)
- ``rams0.5``     — *elastic* PE (RAMS), flat surface, layered elastic bottom

Format reference: ``third_party/ramsurf/readme.orig`` and the upstream
``setup`` subroutines. RAMS swaps row-5's ``ns, rs`` fields for ``irot,
theta`` and adds two profile blocks per range (shear speed + shear
attenuation). RAMSurf inserts a surface ``(range, depth)`` block right
after row 5; ``ramgeo`` uses the base layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from uacpy.core.exceptions import ConfigurationError


#: Block terminator. Every RAM-family block reader stops on a negative first
#: column and never on a count: ``zread`` loops until ``zi.lt.0.0``
#: (``ramsurf/ramsurf1.5.f:205-206``) and the surface / bathymetry loops until
#: ``rsrf(i).lt.0.0`` / ``rb(i).lt.0.0`` (``:83-84``, ``:91-92``). The second
#: column is read but discarded on that row.
_TERM = "-1 -1\n"


def _write_block(
    fh,
    pairs: Sequence[Tuple[float, float]],
) -> None:
    """Write a ``(depth, value)`` block followed by the ``-1 -1`` terminator.

    ``zread`` pins each pair to the grid node ``i = int(1.5 + z/dz)`` and
    linearly interpolates between pinned nodes (``ramsurf1.5.f:202-219``), so
    the block is a control-point list on a ``dz`` grid, not a sampled curve.
    Two pairs closer together than ``dz`` collide on one node; ``:208`` pushes
    the second down a node rather than merging them, which shifts it by up to
    ``dz``. Keep the pairs at least ``dz`` apart to place them exactly.
    """
    if not pairs:
        raise ConfigurationError("Cannot write empty profile block")
    for d, v in pairs:
        fh.write(f"{float(d):.6f} {float(v):.6f}\n")
    fh.write(_TERM)


def write_ramin(
    filepath: Union[str, Path],
    *,
    kind: str,
    fc: float,
    zs: float,
    zr_line: float,
    rmax: float,
    dr: float,
    ndr: int,
    zmax: float,
    dz: float,
    ndz: int,
    zmplt: float,
    c0: float,
    np_pade: int,
    bathymetry: Sequence[Tuple[float, float]],
    range_segments: Sequence[dict],
    surface: Optional[Sequence[Tuple[float, float]]] = None,
    ns_stab: int = 1,
    rs_stab: float = 0.0,
    irot: int = 1,
    theta: float = 60.0,
    title: str = "uacpy ram.in",
) -> None:
    """
    Write a Collins-style ``ram.in``-format file.

    Parameters
    ----------
    filepath : str
        Destination file path. The filename is fixed per binary — each
        hardcodes its own OPEN: ``ramsurf1.5`` reads ``ram.in``
        (``ramsurf1.5.f:31``), ``ramgeo`` reads ``ramgeo.in``
        (``ramgeo1.5.f:62``) and ``rams0.5`` reads ``rams.in`` — so the
        caller must pass the name its target binary opens, in that
        binary's working directory.
    kind : {'rams', 'ramsurf', 'ramgeo'}
        Which binary the file is targeted at. ``'ramsurf'`` adds a
        surface block right after row 5; ``'rams'`` swaps row-5 from
        ``(ns, rs)`` to ``(irot, theta)`` and emits two extra profile
        blocks per range (shear speed + shear attenuation). ``'ramgeo'``
        is the fluid, flat-surface form — row-5 ``(ns, rs)``, no surface
        block, no shear blocks (i.e. ``'ramsurf'`` without the surface).
    fc, zs, zr_line : float
        Centre frequency (Hz), source depth (m), receiver depth (m) at
        which ``tl.line`` is written.
    rmax, dr, ndr : float, float, int
        Domain range (m), range step (m), output stride (every ``ndr``
        steps).
    zmax, dz, ndz, zmplt : float, float, int, float
        Computational depth (m), depth step (m), output stride, plot
        depth (m).
    c0, np_pade : float, int
        Reference sound speed (m/s) and number of Padé coefficients.
    bathymetry : list of (range, depth)
        Seafloor profile vs range, in metres. Linearly interpolated by the
        binary, which self-extends past the last point by repeating its
        depth out to ``2*rmax`` (``ramsurf1.5.f:95-96``,
        ``ramgeo1.5.f:115-116``) or ``rmax + 2*dr`` (``rams0.5.f:116-117``),
        so the profile need not reach ``rmax``.
    range_segments : list of dict
        One entry per range section, in order. The first entry's
        ``range`` is ignored (initial profile); subsequent entries write
        their ``range`` on its own line before the profile blocks. Keys:

        - ``range`` (float, ignored on the first entry)
        - ``water_ssp`` : list of (depth, c)
        - ``bottom_c``  : list of (depth, c) — compressional speed
        - ``bottom_rho``: list of (depth, rho)
        - ``bottom_attn``: list of (depth, attn) — compressional attenuation
        - ``bottom_cs``  (RAMS only): list of (depth, shear speed)
        - ``bottom_attns`` (RAMS only): list of (depth, shear attenuation)
    surface : list of (range, depth), optional
        Surface profile (only used / required when ``kind='ramsurf'``).
        ``depth`` ≥ 0 means how far below z=0 the pressure-release
        surface sits at that range. Self-extends to ``2*rmax`` like the
        bathymetry (``ramsurf1.5.f:87-88``).
    ns_stab, rs_stab : int, float
        Row-5 stability fields (``ramsurf`` only).
    irot, theta : int, float
        Row-5 elastic stability fields (``rams`` only). ``theta`` is the
        Padé rotation angle in degrees (0 < theta < 90).
    title : str
        Header line (row 1). Free text, ignored by the binary.
    """
    kind = kind.lower()
    if kind not in ('rams', 'ramsurf', 'ramgeo'):
        raise ConfigurationError(
            f"kind must be 'rams', 'ramsurf' or 'ramgeo'; got {kind!r}"
        )
    if kind == 'ramsurf' and not surface:
        raise ConfigurationError("kind='ramsurf' requires a surface profile")
    if kind == 'rams':
        for seg in range_segments:
            if 'bottom_cs' not in seg or 'bottom_attns' not in seg:
                raise ConfigurationError(
                    "kind='rams' requires bottom_cs and bottom_attns "
                    "in every range segment"
                )

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as fh:
        # Rows 1-5 go to ``setup``: title, (freq zs zr), (rmax dr ndr),
        # (zmax dz ndz zmplt), then row 5 — (c0 np ns rs) for the fluid
        # codes (``ramsurf1.5.f:76-80``, ``ramgeo1.5.f:104-108``) or
        # (c0 np irot theta) for RAMS (``rams0.5.f:105-109``).
        fh.write(f"{title}\n")
        fh.write(f"{float(fc):.6f} {float(zs):.6f} {float(zr_line):.6f}\n")
        fh.write(f"{float(rmax):.6f} {float(dr):.6f} {int(ndr)}\n")
        fh.write(
            f"{float(zmax):.6f} {float(dz):.6f} {int(ndz)} {float(zmplt):.6f}\n"
        )
        if kind == 'rams':
            fh.write(
                f"{float(c0):.6f} {int(np_pade)} {int(irot)} {float(theta):.6f}\n"
            )
        else:
            fh.write(
                f"{float(c0):.6f} {int(np_pade)} {int(ns_stab)} {float(rs_stab):.6f}\n"
            )

        if kind == 'ramsurf':
            _write_block(fh, surface)

        _write_block(fh, bathymetry)

        # ``profl`` reads a segment's profile blocks first and only then the
        # range at which the *next* segment starts (``rams0.5.f:198``,
        # ``ramsurf1.5.f:180``), defaulting it to ``2*rmax`` at EOF. On disk
        # that puts each range line between the blocks it separates, which is
        # what writing it ahead of every segment but the first produces.
        for i, seg in enumerate(range_segments):
            if i > 0:
                fh.write(f"{float(seg['range']):.6f}\n")
            _write_block(fh, seg['water_ssp'])
            _write_block(fh, seg['bottom_c'])
            if kind == 'rams':
                _write_block(fh, seg['bottom_cs'])
            _write_block(fh, seg['bottom_rho'])
            _write_block(fh, seg['bottom_attn'])
            if kind == 'rams':
                _write_block(fh, seg['bottom_attns'])
