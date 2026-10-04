"""The settings record one :class:`~uacpy.models.sparc.SPARC` run
resolves before it launches."""

from dataclasses import dataclass
from typing import Tuple

from uacpy.core.run_settings import EngineSettings


@dataclass(frozen=True, eq=False)
class SparcSettings(EngineSettings):
    """The settings one :class:`SPARC` run resolved, before launching:
    ``SPARC().run_settings(env, source, receiver).engine``, and
    ``result.run_settings.engine`` on the trace it produced.

    Attributes
    ----------
    pulse_type : str
        The deck's 4-character pulse code. Unpinned, ``'FN+B'`` marches the
        run's ``source_waveform`` (staged as ``STSFIL``) and ``'PN+B'`` the
        canned wavelet when the run was handed no waveform.
    pulse_type_origin : str
        Where ``pulse_type`` came from.
    deck_frequency : float
        The deck's header frequency (Hz), ``source.frequencies[0]``: the
        frequency the seabed and volume attenuation are converted at
        (``sparc.f90:96-97``) and the centre of a canned pulse
        (``sparc.f90:525``).
    freq_min, freq_max : float
        The pulse band (Hz) the deck carries: ``sparc.f90:111-116`` spans
        the wavenumbers ``[2π freq_min / c_high, 2π freq_max / c_low]`` from it.
    freq_min_origin, freq_max_origin : str
        Where each band edge came from: the constructor, ``run(frequencies=)``,
        the base rule's band of the marched waveform padded to the ``time_max``
        record (its -40 dB support, its lowest bin above DC, or the cost
        limit of a hard-edged pulse), or one octave around a canned pulse's
        frequency.
    time_max : float
        The end (s) of the ``[0, time_max]`` output window.
    time_max_origin : str
        Where ``time_max`` came from: the constructor, ``run(output_duration=)``,
        or 2.5 direct travel times to the farthest receiver.
    n_time_samples : int
        The output time samples over ``[0, time_max]``.
    rmax_m : float
        The deck's ``RMax`` (m): the farthest receiver range times
        ``rmax_factor``; it sets the wavenumber step.
    rmax_factor : float
        The margin ``rmax_m`` was built with.
    rmax_factor_origin : str
        Where the margin came from: the constructor or the default.
    c_low, c_high : float
        The phase-speed bounds (m/s) the deck carries.
    n_wavenumbers : int
        The wavenumbers ``sparc.f90:116`` marches.
    n_mesh : tuple of int
        The mesh the deck asks for, one count per medium: the pinned count
        on every medium, or each sized at 20 points per wavelength at
        ``freq_max``.
    output_mode : str
        ``'R'``, ``'D'`` or ``'S'``.
    run_bases : tuple of str
        The deck root of each binary launch, in launch order: one per
        receiver depth (``'R'``), one per receiver range (``'D'``), or one
        (``'S'``).
    march_start, courant_factor : float
        The deck's ``TSTART`` and Courant factor.
    sts_samples : int
        The waveform samples ``STSFIL`` carries (``0``: a canned pulse).
    sts_rows : int
        The rows ``STSFIL`` is written with: the samples, zero-padded to a
        power of two when the pulse is band-passed.
    notices : tuple of Notice
        One per condition of these settings the run warns about: its
        short ``note`` and the ``message`` ``run`` and ``run_settings``
        announce (``validate_inputs`` announces nothing).
    """

    pulse_type: str
    pulse_type_origin: str
    deck_frequency: float
    freq_min: float
    freq_max: float
    freq_min_origin: str
    freq_max_origin: str
    time_max: float
    time_max_origin: str
    n_time_samples: int
    rmax_m: float
    rmax_factor: float
    rmax_factor_origin: str
    c_low: float
    c_high: float
    n_wavenumbers: int
    n_mesh: Tuple[int, ...]
    output_mode: str
    run_bases: Tuple[str, ...]
    march_start: float
    courant_factor: float
    sts_samples: int
    sts_rows: int

    def __post_init__(self):
        # A list (the to_dict form) is stored as the tuple a frozen record
        # holds.
        object.__setattr__(self, 'n_mesh',
                           tuple(int(n) for n in self.n_mesh))
        object.__setattr__(self, 'run_bases', tuple(self.run_bases))
        super().__post_init__()
