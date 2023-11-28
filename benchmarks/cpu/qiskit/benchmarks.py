import numpy as np
import pytest
from qiskit import Aer
from qiskit.algorithms.gradients import ReverseEstimatorGradient
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

np.random.seed(0)


max_parallel_threads = 12
gpu = False
method = "statevector"


def generate_circuit(nqubits):
    ansatz = TwoLocal(
        nqubits,
        ["rx", "ry", "rz"],
        ["cx"],
        "linear",
        reps=3,
        flatten=True,
    ).decompose()
    params = np.random.rand(ansatz.num_parameters)
    return ansatz, params


def execute_statevector(benchmark, circuit, params):
    backend_options = {
        "method": method,
        "precision": "double",
        "max_parallel_threads": max_parallel_threads,
        "fusion_enable": True,
        "fusion_threshold": 14,
        "fusion_max_qubit": 5,
    }

    circuit = circuit.bind_parameters(params)

    backend = Aer.get_backend("statevector_simulator")
    backend.set_options(**backend_options)

    def evalfunc(backend, circuit):
        backend.run(circuit).result()

    benchmark(evalfunc, backend, circuit)


def execute_estimator(benchmark, circuit, obs, params):
    estimator = Estimator()

    def evalfunc(estimator, circuit, obs, params):
        estimator.run([circuit], [obs], [params]).result()

    benchmark(evalfunc, estimator, circuit, obs, params)


def execute_gradient(benchmark, circuit, obs, params):
    estimator_grad = ReverseEstimatorGradient()

    def evalfunc(estimator_grad, circuit, obs, params):
        estimator_grad.run([circuit], [obs], [params]).result()

    benchmark(evalfunc, estimator_grad, circuit, obs, params)


nqubits_list = range(4, 21)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_statevector(benchmark, nqubits):
    benchmark.group = "qiskit_statevector"
    circuit, params = generate_circuit(nqubits)
    execute_statevector(benchmark, circuit, params)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_estimator(benchmark, nqubits):
    benchmark.group = "qiskit_estimator"
    circuit, params = generate_circuit(nqubits)
    obs = SparsePauliOp.from_list([("Z" * nqubits, 1)])
    execute_estimator(benchmark, circuit, obs, params)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_gradient(benchmark, nqubits):
    benchmark.group = "qiskit_gradient"
    circuit, params = generate_circuit(nqubits)
    obs = SparsePauliOp.from_list([("Z" * nqubits, 1)])
    execute_gradient(benchmark, circuit, obs, params)
