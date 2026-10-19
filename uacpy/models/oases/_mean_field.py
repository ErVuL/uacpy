"""The mean-field producer run the scattering programs chain onto (OAST
for OASS, OASP for OASSP), and the two-launch chain both run it in."""

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from uacpy.models.base import PropagationModel, StageInputs
from uacpy.models._launch import require_output
from uacpy.models.oases._base import OASES
from uacpy.core.results import Result
from uacpy.io.oases_reader import OasesRhsHeader, read_oases_rhs_header


class _MeanFieldRun(NamedTuple):
    """What the mean-field launch of an OASSP/OASS call returns: the
    producer's own result (kept as ``components['mean_field']``), the
    header of the ``.rhs`` it wrote, and the stem its files carry."""
    result: Result
    rhs_header: OasesRhsHeader
    stem: str


@dataclass
class _ScatteringChain:
    """What the launches of one OASS or OASSP call share, built once by
    :meth:`_ScatteringPostProcessor._prepare_launches`: the mean-field
    producer, and the scattering process its launch leaves (OASS reads the
    wavenumber count the binary used off its stdout,
    :func:`~uacpy.models.oases.oass._oass_wavenumber_echo`)."""
    producer: PropagationModel
    process: object = None


def _mean_field_inputs(inputs, producer_settings, source):
    """The :class:`~uacpy.models.base.StageInputs` of the mean-field launch:
    the scatterer's work directory and environment, the producer's settings
    for this launch's source depth."""
    return StageInputs(
        work_dir=inputs.work_dir, env=inputs.env, source=source,
        receiver=inputs.receiver,
        settings=producer_settings._replace(
            source_depths=np.atleast_1d(np.asarray(source.depths,
                                                   dtype=float))))


class _ScatteringPostProcessor(OASES):
    """The chain OASS and OASSP run: launch 0 runs the mean-field producer
    through its own stage hooks (deck, binary, result) in this run's work
    directory, where it leaves the ``.rhs`` (FOR045) option ``'s'`` writes;
    launch 1 runs the scattering program on it.

    A subclass gives the producer a call runs (:meth:`_producer_for`), the
    Source the producer runs on (:meth:`_mean_field_source`), the files and
    header checks of the mean field (:meth:`_require_mean_field_files`,
    :meth:`_check_mean_field_header`, :attr:`_RHS_HINT`), and its own deck,
    launch and read (:meth:`_write_scattering_deck`,
    :meth:`_launch_scattering`, :meth:`_read_scattering_output`).
    """

    #: Why a missing or empty ``.rhs`` means what it means, for the refusal.
    _RHS_HINT = ''

    @abstractmethod
    def _producer_for(self, settings) -> PropagationModel:
        """The mean-field producer a call with ``settings`` runs."""

    @abstractmethod
    def _mean_field_source(self, source):
        """The Source the producer runs on, for this call's ``source``."""

    def _require_mean_field_files(self, inputs, deck: Path) -> None:
        """Refuse a mean field missing a file the scattering program opens
        beside the ``.rhs``; none by default."""

    def _check_mean_field_header(self, inputs, header) -> None:
        """Refuse a ``.rhs`` header the scattering run cannot use; none by
        default."""

    @abstractmethod
    def _write_scattering_deck(self, inputs) -> Path:
        """Launch 1's deck."""

    @abstractmethod
    def _launch_scattering(self, inputs, deck: Path) -> None:
        """Launch 1: the scattering binary on its deck."""

    @abstractmethod
    def _read_scattering_output(self, inputs, deck: Path):
        """Launch 1's output, as the reader returns it."""

    def _announce_engine_settings(self, env, source, receiver,
                                  settings) -> None:
        """Stage 3 (run and run_settings): the mean field's notices, then
        the scattering program's own."""
        self._producer_for(settings)._announce_engine_settings(
            env, self._mean_field_source(source), receiver,
            settings.engine.mean_field)
        super()._announce_engine_settings(env, source, receiver, settings)

    # ── stage 4: the mean field, then the scattering program ──────────

    def _n_launches(self, settings) -> int:
        """Two: the mean-field producer, then the scattering program on its
        ``.rhs``."""
        return 2

    def _prepare_launches(self, env, settings):
        """The mean-field producer both launches' hooks call, and the slot
        the scattering launch leaves its process in."""
        return _ScatteringChain(producer=self._producer_for(settings))

    def _mean_field_stage_inputs(self, inputs):
        """The producer's stage inputs for this launch's source depth."""
        return _mean_field_inputs(inputs, inputs.settings.engine.mean_field,
                                  self._mean_field_source(inputs.source))

    def _write_input(self, inputs) -> Path:
        """Stage 4. Launch 0: the mean field's deck, written by the
        producer's own ``_write_input``. Launch 1: the scattering deck
        (:meth:`_write_scattering_deck`)."""
        if inputs.launch == 0:
            producer = inputs.prepared.producer
            self._log(f"Running {producer.model_name} for the mean field "
                      f"(option 's')")
            return producer._write_producer_deck(
                self._mean_field_stage_inputs(inputs))
        return self._write_scattering_deck(inputs)

    def _launch(self, inputs, deck: Path) -> None:
        """Stage 4. Launch 0: the producer's own ``_launch``. Launch 1:
        :meth:`_launch_scattering`."""
        if inputs.launch == 0:
            inputs.prepared.producer._launch_producer(
                self._mean_field_stage_inputs(inputs), deck)
            return
        self._launch_scattering(inputs, deck)

    def _read_output(self, inputs, deck: Path):
        """Stage 4. Launch 0: :meth:`_read_mean_field`. Launch 1:
        :meth:`_read_scattering_output`."""
        if inputs.launch == 0:
            return self._read_mean_field(inputs, deck)
        return self._read_scattering_output(inputs, deck)

    def _read_mean_field(self, inputs, deck: Path) -> _MeanFieldRun:
        """The mean-field launch's output: the producer's own result (its
        stages 4-6, :meth:`~PropagationModel._run_producer_launch`) and the
        header of the ``.rhs`` it wrote, after the files and the header are
        checked (:meth:`_require_mean_field_files`,
        :meth:`_check_mean_field_header`)."""
        result = inputs.prepared.producer._run_producer_launch(
            self._mean_field_stage_inputs(inputs), deck)
        rhs_path = require_output(
            self.model_name, [inputs.work_dir / f'{deck.stem}.045'],
            what="a mean-field .rhs (option 's')", hint=self._RHS_HINT,
        )
        self._require_mean_field_files(inputs, deck)
        header = read_oases_rhs_header(rhs_path)
        self._check_mean_field_header(inputs, header)
        return _MeanFieldRun(result=result, rhs_header=header,
                             stem=deck.stem)
