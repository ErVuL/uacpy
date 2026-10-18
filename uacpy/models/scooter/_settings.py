"""The settings record one :class:`~uacpy.models.scooter.Scooter` run
resolves before it launches."""

from dataclasses import dataclass

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
        deck's ``cLow`` / ``cHigh``. Unpinned, ``c_low`` is the slowest
        speed of every medium (shear speeds and half-spaces included) ×
        0.85 over an elastic half-space, × 0.90 otherwise
        (``scooter.f90:193-259``,
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
    n_mesh : int
        The mesh points per medium the deck asks for (``0``: the binary
        sizes each medium itself).
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
    n_mesh: int
    mesh_reference_frequency: float
    deck_max_frequency: float
    n_wavenumbers: int
    taper: float
    peak_memory_bytes: int
