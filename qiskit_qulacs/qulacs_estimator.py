"""QulacsEstimator class."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.primitives import Estimator
from qiskit.primitives.base import EstimatorResult
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.primitives.utils import _circuit_key, _observable_key, init_observable
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

import qulacs
from qiskit_qulacs.adapter import (
    convert_qiskit_to_qulacs_circuit,
    convert_sparse_pauliop_to_qulacs_obs,
)
from qulacs.circuit import QuantumCircuitOptimizer


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
        self._states = ["QuantumState", "QuantumStateGpu"]
        self.qc_opt = QuantumCircuitOptimizer()

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        # Initialize metadata
        gpu = run_options.pop("gpu", False)
        qco_enable = run_options.pop("qco_enable", False)
        qco_method = run_options.pop("qco_method", "light")
        qco_max_block_size = run_options.pop("qco_max_block_size", 2)

        metadata: list[dict[str, Any]] = [{} for _ in range(len(circuits))]

        bound_circuits = []
        for i, value in zip(circuits, parameter_values):
            if len(value) != len(self._parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self._parameters[i])})."
                )

            bound_circuits.append(self._circuits[i](np.array(value))[0])

        sorted_obs = [self._observables[i] for i in observables]
        expectation_values = np.zeros(len(bound_circuits))

        for i, (circ, obs, metadatum) in enumerate(
            zip(bound_circuits, sorted_obs, metadata)
        ):

            start = time.time()  # Start timer

            state = getattr(qulacs, self._states[gpu])(circ.get_qubit_count())

            if qco_enable:
                if qco_method == "light":
                    self.qc_opt.optimize_light(circ)
                elif qco_method == "greedy":
                    self.qc_opt.optimize(circ, qco_max_block_size)

            circ.update_quantum_state(state)
            expectation_values[i] = obs.get_expectation_value(state)

            metadatum["time_taken"] = time.time() - start  # End timer

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
                self._circuits.append(convert_qiskit_to_qulacs_circuit(circuit))
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
        job._submit()
        return job
