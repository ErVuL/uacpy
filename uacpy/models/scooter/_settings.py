"""The settings record one :class:`~uacpy.models.scooter.Scooter` run
resolves before it launches."""

from dataclasses import dataclass
from typing import Tuple, Union

from uacpy.core.run_settings import EngineSettings


@dataclass(frozen=True, eq=False)
class ScooterSettings(EngineSettings):
    """The settings one :class:`Scooter` run resolved, before launching:
    ``Scooter().run_settings(env, source, receiver).engine``, and
    ``result.run_settings.engine`` on the field it produced.

    Attributes
    ----------
    c_low, c_high : float
        The phase-speed window (m/s) of the wavenumber integration, the
        deck's ``cLow`` / ``cHigh``. Unpinned, ``c_low`` is 0.90 × the
        slowest wave of the problem: every medium's compressional and shear
        speed (half-spaces included) and every Scholte wave
        (``scooter.f90:258``; Brekhovskikh & Lysanov eq. 3.4.29;
        :func:`~uacpy.core.engine_defaults.default_c_low`), and ``c_high``
        is the unbounded ``DEFAULT_C_MAX_UNBOUNDED`` (1e9 m/s, ``k_min = 0``,
        Jensen et al., ch. 4 App. 1) over every seabed.
    c_low_origin, c_high_origin : str
        Where each bound came from: the constructor, or the rule above.
    rmax : float
        The spectral ``RMax`` (m), ``receiver.ranges.max() ×
        rmax_factor``, written in km; it fixes the wavenumber sampling.
    rmax_factor : float
        The multiplier that ``rmax`` was built with.
    rmax_factor_origin : str
        Where ``rmax_factor`` came from: the constructor, or the default
        (2.0 in every run mode, the toolbox's Scooter decks).
    n_mesh : int or tuple of int
        The mesh points the deck asks for: one count for every medium, or
        one per medium, water first (``0``: the binary sizes that medium
        itself).
    mesh_reference_frequency : float
        The deck's header frequency ``freq0`` (Hz): AT checks the mesh
        floor at it, and ``scooter.f90:106-111`` scales the mesh of every
        swept frequency from it, with a 100-point floor per medium.
    deck_max_frequency : float
        ``freqVec(Nfreq)`` (Hz), the frequency ``scooter.f90:67-69`` sizes
        the wavenumber grid at.
    n_wavenumbers : int
        The wavenumber samples ``scooter.f90:69`` derives from this deck.
    taper : float
        The fraction of the wavenumber span rolled off at each edge before
        the Hankel transform (``0``: none). The pass band is placed on the
        phase-speed grid the binary writes into the ``.grn``.
    peak_memory_bytes : int
        The memory one launch reaches at peak (bytes): twice the complex64
        Green's-function cube ``read_grn_file`` allocates, plus the
        transform kernel.
    notices : tuple of Notice
        One per condition of these settings the run warns about: its
        short ``note`` and the ``message`` ``run`` and ``run_settings``
        announce (``validate_inputs`` announces nothing).
    """

    c_low: float
    c_low_origin: str
    c_high: float
    c_high_origin: str
    rmax: float
    rmax_factor: float
    rmax_factor_origin: str
    n_mesh: Union[int, Tuple[int, ...]]
    mesh_reference_frequency: float
    deck_max_frequency: float
    n_wavenumbers: int
    taper: float
    peak_memory_bytes: int

    def __post_init__(self):
        # A list (the to_dict form) is stored as the tuple a frozen record
        # holds.
        if not isinstance(self.n_mesh, int):
            object.__setattr__(self, 'n_mesh',
                               tuple(int(n) for n in self.n_mesh))
        super().__post_init__()
