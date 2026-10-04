"""The settings record one :class:`~uacpy.models.bounce.Bounce` run
resolves before it launches."""

from dataclasses import dataclass
from typing import Optional, Tuple

from uacpy.core.run_settings import EngineSettings


@dataclass(frozen=True, eq=False)
class BounceSettings(EngineSettings):
    """The settings one :class:`Bounce` run resolved, before launching:
    ``Bounce().run_settings(env, source, receiver).engine``, and
    ``result.run_settings.engine`` on the table it produced.

    Attributes
    ----------
    c_low : float
        Smallest phase velocity tabulated (m/s), the deck's ``cLow``.
    c_low_origin : str
        Where ``c_low`` came from: the constructor, or AT ``bounce.htm``'s
        "lowest speed in the problem" read off the environment as given.
    c_high : float
        Largest phase velocity tabulated (m/s), the deck's ``cHigh``.
    c_high_origin : str
        Where ``c_high`` came from: the constructor, or the unbounded
        default ``DEFAULT_C_MAX_UNBOUNDED``.
    rmax_m : float
        The range (m) the table's angular sampling is sized for, written as
        the deck's ``RMax`` in km.
    rmax_origin : str
        Where ``rmax_m`` came from: the constructor, ``n_angles``, the
        receiver's farthest range, or the 10 km fallback.
    n_angles : int
        Angles the binary tabulates for this deck (``bounce.f90:45-49``).
    n_mesh : tuple of int
        Mesh points per sediment medium (empty for a bare half-space).
    staged_table_suffix : str or None
        The reflection table a ``'file'`` seabed stages next to the deck as
        an input of this launch (``'.brc'``), kept out of the pre-launch
        sweep of stale outputs; ``None`` otherwise.
    """

    c_low: float
    c_low_origin: str
    c_high: float
    c_high_origin: str
    rmax_m: float
    rmax_origin: str
    n_angles: int
    n_mesh: Tuple[int, ...]
    staged_table_suffix: Optional[str]

    def __post_init__(self):
        # A list (the to_dict form) is stored as the tuple a frozen record
        # holds.
        object.__setattr__(self, 'n_mesh',
                           tuple(int(n) for n in self.n_mesh))
        super().__post_init__()
