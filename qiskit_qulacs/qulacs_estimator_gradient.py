"""QulacsEstimatorGradient class."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import sympy as sp
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.providers import Options
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.gradients import BaseEstimatorGradient
from qiskit_algorithms.gradients.base.estimator_gradient_result import (
    EstimatorGradientResult,
)

from qiskit_qulacs.adapter import (
    convert_qiskit_to_qulacs_circuit,
    convert_sparse_pauliop_to_qulacs_obs,
)


class QulacsEstimatorGradient(BaseEstimatorGradient):
    """QulacsEstimatorGradient class."""

    def __init__(
        self,
        options: Options | None = None,
    ):
        # this is required by the base class, but not used
        dummy_estimator = Estimator()
        super().__init__(dummy_estimator, options)

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
        metadata: list[dict[str, Any]] = [{} for _ in range(len(circuits))]

        for circuit, observable, parameter_values_, parameters_, metadatum in zip(
            circuits, observables, parameter_values, parameters, metadata
        ):

            qulacs_circuit = convert_qiskit_to_qulacs_circuit(circuit)
            qulacs_obs = convert_sparse_pauliop_to_qulacs_obs(observable)

            start = time.time()  # Start timer

            params_values = np.array(parameter_values_)
            circ, metadata = qulacs_circuit(params_values)
            parameter_mapping = metadata["parameter_mapping"]  # type: ignore
            parameter_exprs = metadata["parameter_exprs"]  # type: ignore

            # Compute gradient using qulacs
            # `np.negative` is used because the gradients computed
            # differ by minus sign
            gradient = np.negative(circ.backprop(qulacs_obs))

            # Evaluate the parameter expressions differentiation and
            # multiply it by the gradient to take into account the
            # expression's differentiation
            for i, idx in enumerate(parameter_mapping):
                f_params, f_expr = parameter_exprs[i]
                f = sp.lambdify(f_params, sp.diff(f_expr))
                gradient[i] = f(params_values[idx]) * gradient[i]

            # The ordering of parameters is changed during circuit conversion.
            # `parameter_mapping` holds this mapping

            # Permute the obtained gradients to match with qiskit's ordering
            gradient[parameter_mapping] = gradient[range(len(parameter_mapping))]

            # Indices of parameters to be differentiated
            indices = [circuit.parameters.data.index(p) for p in parameters_]
            gradients.append(gradient[indices])

            metadatum["time_taken"] = time.time() - start  # End timer

        opt = self._get_local_options(options)
        return EstimatorGradientResult(
            gradients=gradients, metadata=metadata, options=opt
        )
