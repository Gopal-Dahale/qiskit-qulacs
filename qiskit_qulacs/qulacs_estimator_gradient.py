from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.providers import Options
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit.algorithms.gradients import BaseEstimatorGradient
from qiskit.algorithms.gradients.estimator_gradient_result import EstimatorGradientResult
from qiskit_qulacs.adapter import convert_qiskit_to_qulacs_circuit, convert_sparse_pauliop_to_qulacs_observable
from qiskit.algorithms.exceptions import AlgorithmError

class QulacsEstimatorGradient(BaseEstimatorGradient):
    def __init__(
        self,
        estimator: BaseEstimator,
        options: Options | None = None,
    ):
        super().__init__(estimator, options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""

        gradients = []
        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            qulacs_circuit, metadata = convert_qiskit_to_qulacs_circuit(circuit)
            parameter_mapping = metadata['paramater_mapping']
            gradient = np.negative(qulacs_circuit(parameter_values_).backprop(convert_sparse_pauliop_to_qulacs_observable(observable)))

            gradient[parameter_mapping] = gradient[np.arange(len(parameter_mapping))]

            # Indices of parameters to be differentiated
            indices = [circuit.parameters.data.index(p) for p in parameters_]
            gradients.append(gradient[indices])

        opt = self._get_local_options(options)
        return EstimatorGradientResult(gradients=gradients, metadata={}, options=opt)
