"""
QulacsSampler class
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.primitives import Sampler
from qiskit.primitives.base import SamplerResult
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.primitives.utils import _circuit_key
from qiskit.quantum_info import Statevector
from qiskit.result import QuasiDistribution

import qulacs
from qiskit_qulacs.adapter import convert_qiskit_to_qulacs_circuit
from qulacs.circuit import QuantumCircuitOptimizer


class QulacsSampler(Sampler):
    """
    QulacsSampler class.

    :Run Options:
          - **shots** (None or int) --
          The number of shots. If None, it calculates the probabilities.
          Otherwise, it samples from multinomial distributions.

          - **seed** (np.random.Generator or int) --
          Set a fixed seed or generator for the multinomial distribution. If shots is None, this
          option is ignored.
    """

    def __init__(self, *, options: dict | None = None):
        """
        Args:
                  options: Default options.

        Raises:
          QiskitError: if some classical bits are not used for measurements.
        """
        super().__init__(options=options)
        self._states = ["QuantumState", "QuantumStateGpu"]
        self.qc_opt = QuantumCircuitOptimizer()

    def _call(
        self,
        circuits: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        shots = run_options.pop("shots", None)
        seed = run_options.pop("seed", None)
        gpu = run_options.pop("gpu", False)
        qco_enable = run_options.pop("qco_enable", False)
        qco_method = run_options.pop("qco_method", "light")
        qco_max_block_size = run_options.pop("qco_max_block_size", 2)

        if seed is None:
            rng = np.random.default_rng()
        elif isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        # Initialize metadata
        metadata: list[dict[str, Any]] = [{} for _ in range(len(circuits))]

        bound_circuits = []
        qargs_list = []
        for i, value in zip(circuits, parameter_values):
            if len(value) != len(self._parameters[i]):
                raise QiskitError(
                    f"The number of values ({len(value)}) does not match "
                    f"the number of parameters ({len(self._parameters[i])})."
                )
            bound_circuits.append(self._circuits[i](np.array(value))[0])
            qargs_list.append(self._qargs_list[i])

        probabilities = []
        for circ, qargs in zip(bound_circuits, qargs_list):

            state = getattr(qulacs, self._states[gpu])(circ.get_qubit_count())

            if qco_enable:
                if qco_method == "light":
                    self.qc_opt.optimize_light(circ)
                elif qco_method == "greedy":
                    self.qc_opt.optimize(circ, qco_max_block_size)

            circ.update_quantum_state(state)
            probabilities.append(
                Statevector(state.get_vector()).probabilities_dict(
                    qargs=qargs, decimals=16
                )
            )

        if shots is not None:
            for i, prob_dict in enumerate(probabilities):
                counts = rng.multinomial(
                    shots, np.fromiter(prob_dict.values(), dtype=float)
                )
                probabilities[i] = {
                    key: count / shots
                    for key, count in zip(prob_dict.keys(), counts)
                    if count > 0
                }
            for metadatum in metadata:
                metadatum["shots"] = shots

        quasis = [QuasiDistribution(p, shots=shots) for p in probabilities]

        return SamplerResult(quasis, metadata)

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
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
                circuit, qargs = self._preprocess_circuit(circuit)
                self._circuits.append(convert_qiskit_to_qulacs_circuit(circuit))
                self._qargs_list.append(qargs)
                self._parameters.append(circuit.parameters)
        job = PrimitiveJob(self._call, circuit_indices, parameter_values, **run_options)
        job._submit()
        return job
