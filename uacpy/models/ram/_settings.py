"""The settings records of a RAM run: :class:`RamGrid` (one launch's
grid) and :class:`RamSettings` (every launch of a run, the backend and how
their outputs make the result)."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from uacpy.core._records import FrozenRecord
from uacpy.core.run_settings import EngineSettings
from uacpy.models._band import FREQUENCY_ARRAY_FIELDS


@dataclass(frozen=True, eq=False)
class RamGrid(FrozenRecord):
    """The PE grid of one binary launch of a :class:`RAM` run.

    Attributes
    ----------
    frequency : float
        The frequency (Hz) the launch's deck carries: the TL frequency, the
        centre ``fc`` of an mpiramS ``(fc, Q, T)`` sweep, or one bin of a
        Collins band.
    dr, dz : float
        Range and depth steps (m) written to the deck.
    zmax : float
        Depth (m) of the PE domain floor, absorbing layer included, in the
        geometric frame (mpiramS: snapped onto its ``deltaz`` grid).
    n_depth_points : int
        Depth points the binary marches over ``zmax``: mpiramS's
        ``icount = floor(zmax/dz - 0.5) + 2`` (``peramx.f90:391``), the
        Collins codes' ``nz = zmax/dz - 0.5`` (``ramgeo1.5.f:130``).
    theta : float or None
        Padé rotation angle (degrees) a Collins deck carries: rams0.5 reads
        it on row 5, where the fluid codes read their stability terms
        instead. ``None`` on mpiramS.
    ndr : int or None
        Collins output stride: the binary writes every ``ndr``-th range step.
    rmax_march : float or None
        The ``rmax`` a Collins deck carries (m): the first written range at
        or beyond the farthest receiver, less half a step
        (:func:`collins.collins_output_stride`).
    zmplt : float or None
        Collins row-4 ``zmplt`` (m, deck frame): the deepest stored output.
    zr_line : float or None
        Collins ``tl.line`` receiver depth (m, deck frame).
    """

    frequency: float
    dr: float
    dz: float
    zmax: float
    n_depth_points: int
    theta: Optional[float] = None
    ndr: Optional[int] = None
    rmax_march: Optional[float] = None
    zmplt: Optional[float] = None
    zr_line: Optional[float] = None

    _REPR_UNITS = {'frequency': 'Hz', 'dr': 'm', 'dz': 'm', 'zmax': 'm',
                   'theta': '°', 'rmax_march': 'm', 'zmplt': 'm'}

    @classmethod
    def from_dict(cls, d) -> 'RamGrid':
        return cls(**d)

    def summary(self) -> str:
        text = (f"{self.frequency:g} Hz: dr {self.dr:.4g} m, dz {self.dz:.4g} "
                f"m, zmax {self.zmax:.5g} m, {self.n_depth_points} depth "
                f"points")
        if self.theta is not None:
            text += f", theta {self.theta:g} deg"
        if self.ndr is not None:
            text += f", ndr {self.ndr}"
        return text


@dataclass(frozen=True, eq=False)
class RamSettings(EngineSettings):
    """The settings one :class:`RAM` run resolved, before launching:
    ``RAM().run_settings(env, source, receiver).engine``, and
    ``result.run_settings.engine`` on the field it produced.

    Attributes
    ----------
    backend : str
        The binary that runs: ``'mpirams'``, ``'ramgeo'``, ``'rams'`` or
        ``'ramsurf'``.
    backend_origin : str
        ``'RAM(backend=…)'`` when pinned, else what :meth:`RAM.select_backend`
        read off the environment.
    c0 : float
        PE reference speed (m/s) every deck carries.
    c0_origin : str
        ``'RAM(c0=…)'`` or Lytaev (2023) Eq. (15).
    fc, q_factor, record_duration : float or None
        The ``(fc, Q, T)`` sweep the band is marched as (mpiramS reads it
        from ``in.pe``; the Collins loop marches its bins). ``(fc, 1e6, 1)``
        on an mpiramS TL run, whose sweep is the one bin at fc. ``None`` on a
        Collins TL run, and ``q_factor`` / ``record_duration`` ``None`` for a non-uniform Collins
        band, which no sweep describes.
    bandwidth_hz, df_hz : float or None
        Width (Hz) of the marched band and its bin spacing; ``df_hz`` is
        ``None`` for a non-uniform band. ``None`` on a TL run.
    marched_frequencies : ndarray
        Every frequency (Hz) a binary marches.
    requested_frequencies : ndarray or None
        The bins the result carries when they are a subset of the marched
        ones (a requested array inside an ``(fc, Q, T)`` superset); ``None``
        keeps every marched bin.
    zmax_frequency : float
        The frequency (Hz) the domain depth and the absorbing layer are
        sized at: the band's lowest (the longest wavelength), or the TL
        frequency.
    stability_range_m : float or None
        The stability range the deck carries (mpiramS: the pinned value or
        the farthest receiver range; the Collins fluid codes: the pinned
        value or 0, which they expand to ``2 × rmax``). ``None`` on rams0.5,
        whose row 5 carries the rotation instead.
    grids : tuple of RamGrid
        One per binary launch, in launch order: one for mpiramS and a
        Collins TL run, one per bin for a Collins band.
    notices : tuple of Notice
        What resolving these settings found worth saying about the run — a
        grid the budget coarsened, trapped modes it cannot carry, a pinned
        ``dz`` on a node, ... — in the order found; ``run`` and
        ``run_settings`` emit each as a warning of its ``category``,
        ``validate_inputs`` does not.
    """

    backend: str
    backend_origin: str
    c0: float
    c0_origin: str
    fc: Optional[float]
    q_factor: Optional[float]
    record_duration: Optional[float]
    bandwidth_hz: Optional[float]
    df_hz: Optional[float]
    marched_frequencies: np.ndarray
    requested_frequencies: Optional[np.ndarray]
    zmax_frequency: float
    stability_range_m: Optional[float]
    grids: Tuple[RamGrid, ...]

    _ARRAY_FIELDS = FREQUENCY_ARRAY_FIELDS
    _TABLE_FIELD = 'grids'

    def __post_init__(self):
        # A list of dicts (the to_dict form) is stored as the tuple of grids
        # a frozen record holds.
        object.__setattr__(self, 'grids', tuple(
            g if isinstance(g, RamGrid) else RamGrid.from_dict(g)
            for g in self.grids))
        super().__post_init__()

    def summary_lines(self):
        """One ``(name, text)`` pair per field, the grids summarised: one
        line per launch up to three, else the range of each step."""
        out = [(name, text) for name, text in super().summary_lines()
               if name not in ('grids', 'notices')]
        grids = self.grids
        if len(grids) <= 3:
            out.extend((f"grid {i}", g.summary()) for i, g in enumerate(grids))
        else:
            def span(values, unit):
                lo, hi = min(values), max(values)
                return (f"{lo:.4g} {unit}" if lo == hi
                        else f"{lo:.4g}-{hi:.4g} {unit}")
            out.append(('grids', (
                f"{len(grids)} launches, "
                f"{span([g.frequency for g in grids], 'Hz')}: "
                f"dr {span([g.dr for g in grids], 'm')}, "
                f"dz {span([g.dz for g in grids], 'm')}, "
                f"zmax {span([g.zmax for g in grids], 'm')}, "
                f"{max(g.n_depth_points for g in grids)} depth points")))
        out.extend(('notice', text if len(text) <= 100 else text[:97] + '...')
                   for text in (n.message for n in self.notices))
        return out


@dataclass(frozen=True)
class RamKnobs:
    """The knobs of a :class:`RAM` its resolution reads, as one record
    (``RAM._knob_record``, read afresh per call: the attributes can be
    reassigned between runs). Each field is the model attribute of the
    same name, except ``accuracy`` (the resolved Lytaev accuracy) and
    ``accuracy_pinned`` (whether the caller set it)."""
    model_name: str
    backend: Optional[str]
    q_factor: Optional[float]
    record_duration: Optional[float]
    accuracy: float
    accuracy_pinned: bool
    absorber_attenuation: float
    absorber_width_wavelengths: float
    depth_decimation: int
    earth_curvature: bool
    n_pade: int
    n_sediment_points: int
    n_stability: int
    rams_dr_factor: float
    rams_rotation: bool
    rams_rotation_angle: Optional[float]
    stability_range_m: Optional[float]
    angle_max: Optional[float]
    dz: Optional[float]
    dr: Optional[float]
    zmax: Optional[float]
    c0: Optional[float]
