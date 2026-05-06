"""
Writer for the Collins-style ``ram.in`` text input shared by the RAM family
binaries uacpy actually dispatches to:

- ``ramsurf1.5``  — fluid PE, *variable* surface (rough surface / beach)
- ``rams0.5``     — *elastic* PE (RAMS), flat surface, layered elastic bottom

uacpy doesn't build the original Collins ``ram1.5`` (mpiramS handles fluid
+ flat with broadband and range-dependent layered bottom), so this writer
only emits the two formats actually consumed.

Format reference: ``third_party/ramsurf/readme.orig`` and the upstream
``setup`` subroutines. RAMS swaps row-5's ``ns, rs`` fields for ``irot,
theta`` and adds two profile blocks per range (shear speed + shear
attenuation). RAMSurf inserts a surface ``(range, depth)`` block right
after row 5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union


_TERM = "-1 -1\n"


def _write_block(
    fh,
    pairs: Sequence[Tuple[float, float]],
) -> None:
    """Write a ``(depth, value)`` block followed by the ``-1 -1`` terminator."""
    if not pairs:
        raise ValueError("Cannot write empty profile block")
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
    Write a Collins-style ``ram.in`` file.

    Parameters
    ----------
    filepath : str
        Destination file path. Convention is ``ram.in`` in the working
        directory of the binary.
    kind : {'rams', 'ramsurf'}
        Which binary the file is targeted at. ``'ramsurf'`` adds a
        surface block right after row 5; ``'rams'`` swaps row-5 from
        ``(ns, rs)`` to ``(irot, theta)`` and emits two extra profile
        blocks per range (shear speed + shear attenuation).
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
    bathymetry : list of (range_m, depth_m)
        Seafloor profile vs range. Linearly interpolated by the binary.
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
    surface : list of (range_m, depth_m), optional
        Surface profile (only used / required when ``kind='ramsurf'``).
        ``depth`` ≥ 0 means how far below z=0 the pressure-release
        surface sits at that range.
    ns_stab, rs_stab : int, float
        Row-5 stability fields (``ramsurf`` only).
    irot, theta : int, float
        Row-5 elastic stability fields (``rams`` only). ``theta`` is the
        Padé rotation angle in degrees (0 < theta < 90).
    title : str
        Header line (row 1). Free text, ignored by the binary.
    """
    kind = kind.lower()
    if kind not in ('rams', 'ramsurf'):
        raise ValueError(
            f"kind must be 'rams' or 'ramsurf'; got {kind!r}"
        )
    if kind == 'ramsurf' and not surface:
        raise ValueError("kind='ramsurf' requires a surface profile")
    if kind == 'rams':
        for seg in range_segments:
            if 'bottom_cs' not in seg or 'bottom_attns' not in seg:
                raise ValueError(
                    "kind='rams' requires bottom_cs and bottom_attns "
                    "in every range segment"
                )

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as fh:
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
