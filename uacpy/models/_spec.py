"""The declarative manifest of a model class: :class:`ModelSpec`, the
:class:`EngineTraits` it carries, and the capability-flag vocabulary its
``supports`` field is drawn from."""

from dataclasses import dataclass, field, fields
from typing import Optional

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._repr import FieldsRepr
from uacpy.core.run_settings import RunMode
# Source geometries a model may declare via ``ModelSpec.source_types``.
# 'point'  -> AT 'R', cylindrical spreading applied
# 'line'   -> AT 'X', Cartesian spreading
# 'scaled' -> AT 'S', point source with cylindrical spreading removed
# Declared with Source itself, so both layers validate against one set.
from uacpy.core.source import VALID_SOURCE_TYPES
from uacpy.models._projection import (
    DEFAULT_COLLAPSE, VALID_COLLAPSE_METHODS,
)


# Capability-flag names a model may advertise. Each maps to a
# ``_supports_<name>`` instance attribute; the question each answers is "does
# this env *shape* — or this source feature — work with this model?".
# Declaring one is a promise: ``_project_environment`` then leaves that
# feature in the env and emits no warning, so the model's deck writer has to
# carry it — nothing downstream re-checks.
# Keep in lockstep with the ``self._supports_*`` block in
# ``PropagationModel.__init__``.
_CAPABILITY_FLAGS: frozenset = frozenset({
    'altimetry',
    'range_dependent_bathymetry',
    'range_dependent_ssp',
    'range_dependent_bottom',
    'layered_bottom',
    'elastic_media',
    # Derived, like 'volume_attenuation' below: read off
    # ``_NATIVE_MULTI_DEPTH_MODES`` (non-empty -> True), never declared in
    # ``spec.supports``, so the whole-model flag and the per-mode table
    # cannot disagree.
    'multi_source_depth',
    'source_beam_pattern',
    'rough_surface',
    'rough_bottom',
    # Declarative ONLY: unlike every other flag this one drives no branch in
    # ``_project_environment`` -- ``env.absorption`` passes through untouched
    # -- and it is NOT read from ``spec.supports``. It mirrors
    # ``spec.traits.consumes_volume_absorption`` so there is one source of
    # truth for "does this engine honour env.absorption", the question OASR
    # and Bounce answer no to.
    'volume_attenuation',
})


#: Modes that consume exactly one source frequency. A multi-frequency Source
#: passed to one of these is a configuration error — the user should pick
#: BROADBAND/TIME_SERIES, or REFLECTION/COVARIANCE/REPLICA for the OASES
#: family that genuinely supports multi-freq sweeps.
_SINGLE_FREQUENCY_MODES: frozenset = frozenset({
    RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.SEMICOHERENT_TL,
    RunMode.RAYS, RunMode.EIGENRAYS, RunMode.ARRIVALS, RunMode.MODES,
})


@dataclass(frozen=True)
class EngineTraits:
    """What an engine does with the parts of a call the base class decides:
    which run keywords it reads, which source-depth stacks it builds itself,
    which band notices it gives, how deep its field reaches, and which of its
    binary's fatal messages describe a result. Declared once per class as
    ``spec.traits``; a spec without traits takes these defaults.

    Fields
    ------
    consumes_volume_absorption : bool
        The engine carries ``env.absorption`` through to its deck. Off by
        default, so an engine that ignores volume attenuation (Bounce, OASR,
        SPARC) cannot advise a user to set it.
    none_receiver_modes : frozenset of RunMode
        The run modes where ``run(env, source, None)`` is legal: an output
        with no receiver geometry — Bounce's reflection table, where
        ``rmax_m=`` can stand in, and Kraken's MODES, whose depth grid the
        engine builds itself. Any other mode refuses ``None`` as the wrong
        carrier type.
    consumes_run_t_start : bool
        A TIME_SERIES run places its output record where
        ``run(t_start=...)`` asks: the engines that synthesise through
        ``_finish_broadband`` or Bellhop's delay-and-sum.
    consumes_single_mode_frequencies : bool
        ``run(frequencies=...)`` means something on a single-frequency run
        mode: OASP, whose solver always runs a frequency sweep that the
        value pins. Every other engine takes a single-frequency run's
        frequency from ``source.frequencies`` alone, so ``run()`` refuses the
        keyword there.
    time_series_ignores_frequencies : str or None
        ``None`` when a TIME_SERIES run reads ``frequencies=``; otherwise
        what the engine does instead, which the warning dropping the keyword
        quotes (Bellhop, whose delay-and-sum labels its trace with the grid
        the pulse implies).
    pads_pulse_to_output_duration : bool
        ``run(output_duration=...)`` zero-pads the pulse the time settings
        record, as the IFFT synthesis reads it. False for SPARC, whose
        ``output_duration`` is the end of its own record (``time_max``) and
        whose STSFIL carries the pulse unpadded.
    receivers_reach_sediment : bool
        The engine resolves the field THROUGH the sediment layers it is
        given (Kraken, Scooter, SPARC, OASES mesh them as media), so a
        receiver — or a buried source — inside the seabed is a supported
        geometry down to the deepest interface. Ray models stop at the
        seafloor; RAM keeps the seafloor by its own override even though its
        PE marches deeper (``RAM._max_receiver_depth``).
    native_multi_depth_modes : frozenset of RunMode
        Modes in which the engine writes every source depth into one deck
        and its reader splits the output into a ``ResultStack``; the base's
        per-depth loop stands aside for these.
    python_stacked_modes : frozenset of RunMode
        Modes outside the field modes in which the engine returns a
        ``ResultStack`` over the source depths anyway, by looping in Python
        rather than in one deck (Bellhop's eigenrays, whose ``.ray`` file
        carries no per-source boundary). Legal input, just not one launch.
    announced_band_modes : frozenset of RunMode
        The modes whose band, derived by the base rules, the engine
        propagates, so stage 3 announces it: the TIME_SERIES grid derived
        from the pulse, the 1 Hz floor of a one-carrier BROADBAND band. An
        engine that derives or runs its band otherwise (a time-marching
        record, a delay-and-sum labelled by the grid, an FFT ladder of its
        own) leaves the mode out; the refusal of a pulse with no usable band
        applies whatever this says.
    single_frequency_modes : frozenset of RunMode
        Modes that consume exactly one source frequency
        (:data:`_SINGLE_FREQUENCY_MODES` unless the engine declares more).
    benign_fortran_fatals : tuple of str
        ERROUT messages that describe a physical outcome the wrapper models
        explicitly, rather than a run failure. Which messages are benign is
        solver-specific, so each engine declares its own.
    """

    consumes_volume_absorption: bool = False
    none_receiver_modes: frozenset = frozenset()
    consumes_run_t_start: bool = False
    consumes_single_mode_frequencies: bool = False
    time_series_ignores_frequencies: Optional[str] = None
    pads_pulse_to_output_duration: bool = True
    receivers_reach_sediment: bool = False
    native_multi_depth_modes: frozenset = frozenset()
    python_stacked_modes: frozenset = frozenset()
    announced_band_modes: frozenset = frozenset({RunMode.TIME_SERIES,
                                                 RunMode.BROADBAND})
    single_frequency_modes: frozenset = _SINGLE_FREQUENCY_MODES
    benign_fortran_fatals: tuple = ()

    def validate(self, model_name: str) -> None:
        """Fail loudly at class-definition time on malformed traits: a mode
        set holding anything but :class:`RunMode` members, or a message
        list that is not a tuple of strings."""
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name.endswith('_modes'):
                if not isinstance(value, frozenset) or not all(
                        isinstance(m, RunMode) for m in value):
                    raise TypeError(
                        f"{model_name}.spec.traits.{f.name} must be a "
                        f"frozenset of RunMode members; got {value!r}.")
        if not (isinstance(self.benign_fortran_fatals, tuple) and all(
                isinstance(m, str) for m in self.benign_fortran_fatals)):
            raise TypeError(
                f"{model_name}.spec.traits.benign_fortran_fatals must be a "
                f"tuple of strings; got {self.benign_fortran_fatals!r}.")


@dataclass(frozen=True)
class ModelSpec(FieldsRepr):
    """Declarative per-model metadata read by :class:`PropagationModel`.

    Consolidates the *static* facts about a model — the run modes it
    emits, the environment shapes it handles natively, its physics-aware
    collapse defaults, and how to locate its binary — into one block the
    base class reads and **validates at class-definition time**, instead
    of scattering them across ``__init__``. The model stays a
    ``PropagationModel`` subclass, so all generic machinery (collapse
    application, validation, file manager, subprocess, ``copy()``) is
    inherited unchanged; the spec only supplies metadata.

    Precedence is preserved: ``collapse`` here layers on top of
    :data:`DEFAULT_COLLAPSE` but never overrides an explicit
    ``Model(collapse={...})`` user value (same rule as
    :meth:`PropagationModel._set_collapse_defaults`). A subclass may still
    set an instance-dependent flag in ``__init__`` *after* ``super().__init__``
    for the rare case a capability depends on a constructor argument.

    Fields
    ------
    modes : sequence of RunMode
        Run modes the model emits. Becomes ``self._supported_modes``; the
        first entry is the default when ``run_mode=None`` unless the model
        overrides ``_default_run_mode`` (Kraken, whose list opens with
        ``MODES`` while its ``run`` defaults to ``COHERENT_TL``).
    supports : iterable of str
        Capability-flag names (subset of :data:`_CAPABILITY_FLAGS`) the
        model honours natively. Every flag not listed defaults ``False``
        and its env feature is collapsed by ``_project_environment``.
    source_types : frozenset of str
        Source geometries (subset of :data:`VALID_SOURCE_TYPES`) the model
        honours. Becomes ``self._supported_source_types``; a ``Source``
        carrying anything else is rejected by ``validate_inputs``.
    collapse : dict
        Per-model collapse defaults overriding :data:`DEFAULT_COLLAPSE`.
    traits : EngineTraits
        What the engine does with the parts of a call the base decides
        (:class:`EngineTraits`); the defaults when not declared.

    Provenance/licence is intentionally *not* here: it is an orthogonal axis
    (who wrote the engine, under what licence) from the run-behaviour fields
    above, and the codebase already keeps dataset provenance in a separate
    catalogue (:mod:`uacpy.data.sources`) rather than folding it into the
    carriers. Models declare it the same way — a ``source`` class attribute
    referencing :data:`uacpy.models.provenance.MODEL_PROVENANCE`.

    Binary resolution is intentionally *not* here either: nothing generic reads it,
    its shape differs per model (single name vs. list of search dirs vs. the
    OASES helper vs. Bellhop's backend dispatch), and multi-binary models
    (Kraken's krakenc, RAM's Collins backends) pick the real executable at
    ``run()`` time. Each model resolves ``self._exe`` in its own ``__init__``.
    """

    modes: tuple = ()
    supports: frozenset = frozenset()
    source_types: frozenset = frozenset({'point'})
    collapse: dict = field(default_factory=dict)
    traits: EngineTraits = EngineTraits()

    _REPR_FIELDS = ('modes', 'supports', 'source_types')

    def validate(self, model_name: str) -> None:
        """Fail loudly at class-definition time on a malformed spec.

        Parameters
        ----------
        model_name : str
            The class the spec belongs to, named in the refusal.
        """
        for m in self.modes:
            if not isinstance(m, RunMode):
                raise TypeError(
                    f"{model_name}.spec.modes must contain RunMode members; "
                    f"got {m!r}."
                )
        bad_flags = set(self.supports) - _CAPABILITY_FLAGS
        if bad_flags:
            raise ConfigurationError(
                f"{model_name}.spec.supports has unknown capability flags: "
                f"{sorted(bad_flags)}. Valid: {sorted(_CAPABILITY_FLAGS)}."
            )
        bad_types = set(self.source_types) - VALID_SOURCE_TYPES
        if bad_types:
            raise ConfigurationError(
                f"{model_name}.spec.source_types has unknown geometries: "
                f"{sorted(bad_types)}. Valid: {sorted(VALID_SOURCE_TYPES)}."
            )
        if not self.source_types:
            raise ConfigurationError(
                f"{model_name}.spec.source_types is empty; every model must "
                f"accept at least one source geometry."
            )
        unknown = set(self.collapse) - set(DEFAULT_COLLAPSE)
        if unknown:
            raise ConfigurationError(
                f"{model_name}.spec.collapse has unknown keys: "
                f"{sorted(unknown)}. Valid keys: {sorted(DEFAULT_COLLAPSE)}."
            )
        for key, value in self.collapse.items():
            if value not in VALID_COLLAPSE_METHODS[key]:
                raise ConfigurationError(
                    f"{model_name}.spec.collapse[{key!r}] = {value!r} is "
                    f"invalid. Valid: {sorted(VALID_COLLAPSE_METHODS[key])}."
                )
        if not isinstance(self.traits, EngineTraits):
            raise TypeError(
                f"{model_name}.spec.traits must be an EngineTraits, got "
                f"{type(self.traits).__name__}.")
        self.traits.validate(model_name)
