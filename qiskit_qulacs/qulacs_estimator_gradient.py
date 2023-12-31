"""QulacsEstimatorGradient class."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit.algorithms.gradients import BaseEstimatorGradient
from qiskit.algorithms.gradients.base.estimator_gradient_result import (
    EstimatorGradientResult,
)
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseEstimator
from qiskit.providers import Options
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit_qulacs.adapter import (
    convert_qiskit_to_qulacs_circuit,
    convert_sparse_pauliop_to_qulacs_obs,
)


class QulacsEstimatorGradient(BaseEstimatorGradient):
    """QulacsEstimatorGradient class."""

    def __init__(
        self,
        estimator: BaseEstimator,
        options: Options | None = None,
    ):
        super().__init__(estimator, options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | SparsePauliOp],
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
            qulacs_obs = convert_sparse_pauliop_to_qulacs_obs(observable)
            parameter_mapping = metadata["paramater_mapping"]
            gradient = np.negative(
                qulacs_circuit(np.array(parameter_values_)).backprop(qulacs_obs)
            )

            gradient[parameter_mapping] = gradient[np.arange(len(parameter_mapping))]

            # Indices of parameters to be differentiated
            indices = [circuit.parameters.data.index(p) for p in parameters_]
            gradients.append(gradient[indices])

        opt = self._get_local_options(options)
        return EstimatorGradientResult(gradients=gradients, metadata={}, options=opt)
