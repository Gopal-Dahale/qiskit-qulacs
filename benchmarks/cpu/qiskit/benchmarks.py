import numpy as np
import pytest
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer
from qiskit_algorithms.gradients import ReverseEstimatorGradient

np.random.seed(0)
nqubits_list = range(4, 21)
nlayers = 3

# naming convention for benchmark group: libname_testname (only one underscore)


def generate_circuit(nqubits):
    ansatz = TwoLocal(
        nqubits,
        ["rx", "ry", "rz"],
        ["cx"],
        "linear",
        reps=nlayers,
        flatten=True,
    )
    params = np.random.rand(ansatz.num_parameters)
    return ansatz, params


def execute_statevector(benchmark, circuit, params):
    backend_options = {
        "method": "statevector",
        "precision": "double",
        "max_parallel_threads": 12,
        "fusion_enable": True,
        "fusion_threshold": 14,
        "fusion_max_qubit": 5,
    }

    circuit = circuit.assign_parameters(params)
    circuit.save_statevector()

    backend = Aer.get_backend("aer_simulator_statevector")
    backend.set_options(**backend_options)

    def evalfunc(backend, circuit):
        sv = backend.run(circuit).result().get_statevector()

    benchmark(evalfunc, backend, circuit)


def execute_estimator(benchmark, circuit, obs, params):
    estimator = Estimator()

    def evalfunc(estimator, circuit, obs, params):
        expval = estimator.run([circuit], [obs], [params]).result().values[0]

    benchmark(evalfunc, estimator, circuit, obs, params)


def execute_sampler(benchmark, circuit, params):
    sampler = Sampler()

    def evalfunc(sampler, circuit, params):
        quasi_dists = sampler.run([circuit], [params]).result().quasi_dists

    benchmark(evalfunc, sampler, circuit, params)


def execute_estgradient(benchmark, circuit, obs, params):
    estimator_grad = ReverseEstimatorGradient()

    def evalfunc(estimator_grad, circuit, obs, params):
        grads = estimator_grad.run([circuit], [obs], [params]).result().gradients[0]

    benchmark(evalfunc, estimator_grad, circuit, obs, params)


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
def test_sampler(benchmark, nqubits):
    benchmark.group = "qiskit_sampler"
    circuit, params = generate_circuit(nqubits)
    circuit.measure_all()
    execute_sampler(benchmark, circuit, params)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_estgradient(benchmark, nqubits):
    benchmark.group = "qiskit_estgradient"
    circuit, params = generate_circuit(nqubits)
    obs = SparsePauliOp.from_list([("Z" * nqubits, 1)])
    execute_estgradient(benchmark, circuit, obs, params)
