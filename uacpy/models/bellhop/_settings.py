"""The settings record one :class:`~uacpy.models.bellhop.Bellhop` run
resolves before it launches, and the values its ``band_origin`` takes for a
band the caller chose."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from uacpy.core.run_settings import EngineSettings, RunSettings

#: ``BellhopSettings.band_origin`` of a BROADBAND grid the caller chose: a
#: ``frequencies=`` passed to the run, or a multi-frequency Source that is
#: the band. Any other origin is the package's expansion of one carrier.
_BAND_FROM_CALL = 'frequencies='
_BAND_FROM_SOURCE = 'Source.frequencies'


@dataclass(frozen=True, eq=False)
class BellhopSettings(EngineSettings):
    """The settings one :class:`Bellhop` run resolved, before launching:
    ``Bellhop().run_settings(env, source, receiver).engine``, and
    ``result.run_settings.engine`` on the result it produced.

    Attributes
    ----------
    backend : str
        The binary the run launches: ``'fortran'``, ``'cxx'``, ``'cuda'``,
        or ``'custom'`` for a pinned ``executable=`` whose name matches none.
        An auto-selected bellhopcuda that finds no CUDA device moves to the
        next installed binary at launch; ``result.backend`` names the one
        that ran.
    executable : str or None
        The file name of a pinned binary whose name matches no build
        (``backend`` ``'custom'``); ``None`` for a named build, which
        ``backend`` names. The path is not recorded: it is the machine's,
        as every other engine's binary path is (``_HOST_KNOBS``).
    run_type : str
        The deck's ``RunType(1:1)``: ``'C'``, ``'I'``, ``'S'`` (TL),
        ``'R'`` (rays), ``'E'`` (eigenrays) or ``'A'`` (arrivals, which the
        BROADBAND and TIME_SERIES syntheses are built from).
    fc : float
        The one frequency (Hz) the rays are traced at: the source frequency,
        or on BROADBAND / TIME_SERIES the centre of a multi-frequency
        Source's band.
    beam_type, n_beams, launch_angles_deg
        ``RunType(2:2)``, the beam count and the launch-angle limits (deg),
        as written to the deck. ``n_beams`` 0 is the binary's automatic
        count (``angleMod.f90:38-51``: 50 rays on a RAYS run, else the
        larger of ``0.3·r_max·f/c0``, 300 and ``π / atan(D/(10·r_max))``),
        which the binary decides at run time and prints to its ``.prt``
        ("Number of beams in elevation", ``angleMod.f90:70``); this record
        does not hold it, the result's ``run_settings.notes`` does.
    ray_step : float
        The ray step (m) the deck carries.
    ray_step_origin : str
        Where ``ray_step`` came from: the constructor, ``env.depth / 50``,
        or ``2 × r_box / MaxN`` for the selected build.
    z_box, r_box : float
        The ray box (m) the deck carries: a ray is dropped once its depth
        exceeds ``z_box`` or its range exceeds ``r_box``.
    z_box_origin, r_box_origin : str
        Where each came from: the constructor, or ``1.2 ×`` the seafloor
        depth / the receiver range extent (10 km for an all-zero range
        axis).
    interp_ssp : str
        The SSP interpolation the deck carries: the constructor's, or for
        ``None`` the automatic choice, ``'quad'`` for a range-dependent
        profile and ``'linear'`` otherwise.
    beam_width_type, beam_curvature, eps_multiplier, r_loop, n_image, ib_win
        The Cerveny beam values the deck's two extra beam lines carry for
        ``beam_type`` 'C' or 'R' (two letters, a float, ``r_loop`` in
        metres, two counts): the constructor's, or for ``None``
        :data:`~uacpy.io.bellhop_writer.CERVENY_DEFAULTS`. ``None`` for
        every other beam type, whose deck carries none of them.
    deck_ranges : ndarray or None
        The receiver ranges (m) the deck carries when they differ from the
        caller's: one extra range at each end for a beam type that never
        fills the first (or, ``'C'``, the last) column.
    range_trim : tuple of int or None
        ``(lo, hi)``: the slice of the deck's range axis that is the
        caller's.
    band_origin : str or None
        BROADBAND only: where the frequency grid came from.
    pulse_samples : int or None
        TIME_SERIES only: the length of the source pulse as handed, before
        the zero padding to ``output_duration``; the delay-and-sum convolves
        these samples.
    bounce : RunSettings or None
        The settings of the BOUNCE run whose reflection table replaces the
        seabed, or ``None`` when the deck carries the seabed itself.
    bounce_origin : str or None
        Why the seabed goes through BOUNCE: ``'run_with_bounce'``, or
        ``'auto_bounce'`` for a layered bottom.
    bounce_c_low, bounce_c_high, bounce_rmax : float or None
        The BOUNCE knobs of that route (``None``: BOUNCE's own default).
    notices : tuple of Notice
        The warnings ``run`` and ``run_settings`` announce about how the
        run will go (``validate_inputs`` announces nothing).
    """

    backend: str
    executable: Optional[str]
    run_type: str
    fc: float
    beam_type: str
    n_beams: int
    launch_angles_deg: Tuple[float, float]
    ray_step: float
    ray_step_origin: str
    z_box: float
    z_box_origin: str
    r_box: float
    r_box_origin: str
    interp_ssp: str
    beam_width_type: Optional[str]
    beam_curvature: Optional[str]
    eps_multiplier: Optional[float]
    r_loop: Optional[float]
    n_image: Optional[int]
    ib_win: Optional[int]
    deck_ranges: Optional[np.ndarray]
    range_trim: Optional[Tuple[int, int]]
    band_origin: Optional[str]
    pulse_samples: Optional[int]
    bounce: Optional[RunSettings]
    bounce_origin: Optional[str]
    bounce_c_low: Optional[float]
    bounce_c_high: Optional[float]
    bounce_rmax: Optional[float]

    _ARRAY_FIELDS = ('deck_ranges',)

    def __post_init__(self):
        # The to_dict form holds lists; the frozen record holds tuples.
        object.__setattr__(self, 'launch_angles_deg',
                           tuple(float(a) for a in self.launch_angles_deg))
        if self.range_trim is not None:
            object.__setattr__(self, 'range_trim',
                               tuple(int(i) for i in self.range_trim))
        super().__post_init__()

    @classmethod
    def from_dict(cls, d) -> 'BellhopSettings':
        d = dict(d)
        if d.get('bounce') is not None:
            d['bounce'] = RunSettings.from_dict(d['bounce'])
        return super().from_dict(d)
