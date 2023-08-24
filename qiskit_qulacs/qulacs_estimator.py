"""QulacsEstimator class."""
from __future__ import annotations

import typing
from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.primitives import Estimator
from qiskit.primitives.base import EstimatorResult
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.primitives.utils import _circuit_key, _observable_key, init_observable
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qulacs import QuantumState

from qiskit_qulacs.adapter import (
    convert_qiskit_to_qulacs_circuit,
    convert_sparse_pauliop_to_qulacs_obs,
)

if typing.TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp


class QulacsEstimator(Estimator):
    """QulacsEstimator class."""

    def __init__(self, *, options: dict | None = None):
        """
        Args:
        options: Default options.

        Raises:
            QiskitError: if some classical bits are not used for measurements.
        """
        super().__init__(options=options)
        self._circuit_ids = {}  # type: ignore
        self._observable_ids = {}  # type: ignore

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        # Initialize metadata
        metadata: list[dict[str, Any]] = [{} for _ in range(len(circuits))]

        bound_circuits = []
        for i, value in zip(circuits, parameter_values):
            if len(value) != len(self._parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self._parameters[i])})."
                )

            bound_circuits.append(self._circuits[i](np.array(value)))

        sorted_obs = [self._observables[i] for i in observables]
        expectation_values = []

        for circ, obs, _ in zip(bound_circuits, sorted_obs, metadata):
            state = QuantumState(circ.get_qubit_count())
            circ.update_quantum_state(state)
            expectation_value = obs.get_expectation_value(state)
            expectation_values.append(expectation_value)

        return EstimatorResult(np.real_if_close(expectation_values), metadata)

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[BaseOperator | SparsePauliOp, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ):
        circuit_indices = []
        for circuit in circuits:
            key = _circuit_key(circuit)
            index = self._circuit_ids.get(key)
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[key] = len(self._circuits)
                self._circuits.append(convert_qiskit_to_qulacs_circuit(circuit)[0])
                self._parameters.append(circuit.parameters)

        observable_indices = []
        for observable in observables:
            observable = init_observable(observable)
            index = self._observable_ids.get(_observable_key(observable))
            if index is not None:
                observable_indices.append(index)
            else:
                observable_indices.append(len(self._observables))
                self._observable_ids[_observable_key(observable)] = len(
                    self._observables
                )
                self._observables.append(
                    convert_sparse_pauliop_to_qulacs_obs(observable)
                )

        job = PrimitiveJob(
            self._call,
            circuit_indices,
            observable_indices,
            parameter_values,
            **run_options,
        )
        job.submit()
        return job
