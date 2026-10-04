"""A result that is not what its run's settings declare is refused with
OutputContractError, a UACPYError: the check every run passes through
(check_output_contract) raises it naming the engine, the mode and each
mismatch, and passes a result that keeps the contract."""
import numpy as np
import pytest

import uacpy
from uacpy.core.results import Field, ReflectionCoefficient
from uacpy.core.run_settings import OutputSpec, RunMode, RunSettings
from uacpy.models._extract import check_output_contract


def _settings():
    return RunSettings(model='Stub', mode=RunMode.COHERENT_TL,
                       frequencies=np.array([100.0]),
                       source_depths=np.array([5.0]),
                       output=OutputSpec(result_type='Field', kind='pressure',
                                         unit='Pa'))


def _field(unit='Pa'):
    return Field(data=np.ones((1, 2), dtype=complex) if unit == 'Pa'
                 else np.ones((1, 2)),
                 coords={'depth': np.array([10.0]),
                         'range': np.array([100.0, 200.0])},
                 kind='pressure', unit=unit)


def test_it_is_a_uacpy_error_exported_at_the_top():
    assert issubclass(uacpy.OutputContractError, uacpy.UACPYError)


def test_a_result_that_keeps_the_contract_passes():
    check_output_contract('Stub', _field(), _settings())


def test_a_wrong_unit_is_refused_naming_engine_mode_and_mismatch():
    with pytest.raises(uacpy.OutputContractError,
                       match=r"Stub\._to_result broke the output contract of "
                             r"COHERENT_TL: unit 'dB', declared 'Pa'\."):
        check_output_contract('Stub', _field('dB'), _settings())


def test_a_wrong_result_class_is_refused():
    table = ReflectionCoefficient(angles=[10.0, 20.0], magnitude=[0.9, 0.8],
                                  phase=[0.0, 0.1])
    with pytest.raises(uacpy.OutputContractError,
                       match='result class ReflectionCoefficient, declared '
                             'Field'):
        check_output_contract('Stub', table, _settings())
