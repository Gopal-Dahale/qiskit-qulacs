import numpy as np
import pytest
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import BackendEstimator, BackendSampler
from qiskit.quantum_info import SparsePauliOp

from qiskit_qulacs import QulacsProvider
from qiskit_qulacs.qulacs_estimator import QulacsEstimator
from qiskit_qulacs.qulacs_estimator_gradient import QulacsEstimatorGradient
from qiskit_qulacs.qulacs_sampler import QulacsSampler

np.random.seed(0)
nqubits_list = range(4, 21)
nlayers = 3


def generate_circuit(nqubits):
    ansatz = TwoLocal(
        nqubits,
        ["rx", "ry", "rz"],
        ["cx"],
        "linear",
        reps=nlayers,
        flatten=False,
    ).decompose()
    params = np.random.rand(ansatz.num_parameters)
    return ansatz, params


def execute_statevector(benchmark, circuit, params):
    backend_options = {
        "shots": 0,
        "device": "CPU",
        "qco_enable": True,
        "qco_method": "light",
    }

    circuit = circuit.assign_parameters(params)

    backend = QulacsProvider().get_backend("qulacs_simulator")
    backend.set_options(**backend_options)

    def evalfunc(backend, circuit):
        sv = backend.run(circuit).result().get_statevector()

    benchmark(evalfunc, backend, circuit)


def execute_estimator(benchmark, circuit, obs, params):
    est_options = {
        "qco_enable": True,
        "qco_method": "light",
    }
    estimator = QulacsEstimator(options=est_options)

    def evalfunc(estimator, circuit, obs, params):
        expval = estimator.run([circuit], [obs], [params]).result().values[0]

    benchmark(evalfunc, estimator, circuit, obs, params)


def execute_backendestimator(benchmark, circuit, obs, params):
    backend_options = {
        "shots": 0,
        "device": "CPU",
        "qco_enable": True,
        "qco_method": "light",
    }

    backend = QulacsProvider().get_backend("qulacs_simulator")
    backend.set_options(**backend_options)

    backend_estimator = BackendEstimator(
        backend=backend, abelian_grouping=False, skip_transpilation=True
    )

    def evalfunc(backend_estimator, circuit, obs, params):
        expval = backend_estimator.run([circuit], [obs], [params]).result().values[0]

    benchmark(evalfunc, backend_estimator, circuit, obs, params)


def execute_sampler(benchmark, circuit, params):
    sampler_options = {
        "qco_enable": True,
        "qco_method": "light",
    }
    sampler = QulacsSampler(options=sampler_options)

    def evalfunc(sampler, circuit, params):
        quasi_dists = sampler.run([circuit], [params]).result().quasi_dists

    benchmark(evalfunc, sampler, circuit, params)


def execute_backendsampler(benchmark, circuit, params):
    backend_options = {
        "shots": 0,
        "device": "CPU",
        "qco_enable": True,
        "qco_method": "light",
    }

    backend = QulacsProvider().get_backend("qulacs_simulator")
    backend.set_options(**backend_options)

    backend_sampler = BackendSampler(backend=backend, skip_transpilation=True)

    def evalfunc(backend_sampler, circuit, params):
        quasi_dists = backend_sampler.run([circuit], [params]).result().quasi_dists

    benchmark(evalfunc, backend_sampler, circuit, params)


def execute_estgradient(benchmark, circuit, obs, params):

    estimator_grad = QulacsEstimatorGradient()

    def evalfunc(estimator_grad, circuit, obs, params):
        grads = estimator_grad.run([circuit], [obs], [params]).result().gradients[0]

    benchmark(evalfunc, estimator_grad, circuit, obs, params)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_statevector(benchmark, nqubits):
    benchmark.group = "qiskit_qulacs_statevector"
    circuit, params = generate_circuit(nqubits)
    execute_statevector(benchmark, circuit, params)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_estimator(benchmark, nqubits):
    benchmark.group = "qiskit_qulacs_estimator"
    circuit, params = generate_circuit(nqubits)
    obs = SparsePauliOp.from_list([("Z" * nqubits, 1)])
    execute_estimator(benchmark, circuit, obs, params)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_backendestimator(benchmark, nqubits):
    benchmark.group = "qiskit_qulacs_backendestimator"
    circuit, params = generate_circuit(nqubits)
    obs = SparsePauliOp.from_list([("Z" * nqubits, 1)])
    execute_backendestimator(benchmark, circuit, obs, params)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_sampler(benchmark, nqubits):
    benchmark.group = "qiskit_qulacs_sampler"
    circuit, params = generate_circuit(nqubits)
    circuit.measure_all()
    execute_sampler(benchmark, circuit, params)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_backendsampler(benchmark, nqubits):
    benchmark.group = "qiskit_qulacs_backendsampler"
    circuit, params = generate_circuit(nqubits)
    circuit.measure_all()
    execute_backendsampler(benchmark, circuit, params)


@pytest.mark.parametrize("nqubits", nqubits_list)
def test_estgradient(benchmark, nqubits):
    benchmark.group = "qiskit_qulacs_gradient"
    circuit, params = generate_circuit(nqubits)
    obs = SparsePauliOp.from_list([("Z" * nqubits, 1)])
    execute_estgradient(benchmark, circuit, obs, params)
