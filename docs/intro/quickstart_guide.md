# Quickstart Guide

Welcome to the Quickstart Guide for Qiskit-Qulacs! This guide will help you quickly get started with using Qiskit-Qulacs as a backend for Qiskit and exploring its functionality.

## Installation

See [Installation Guide](INSTALL.md) for instructions.

## Usage

Once you have installed the necessary dependencies, you can start using Qiskit-Qulacs. Here's a brief overview of the main components and their usage:

###  QulacsProvider
The QulacsProvider uses qiskit's Provider Interface and allows you to perform local statevector simulations with Qulacs as the backend. You can import and initialize it as follows:

```python
from qiskit_qulacs import QulacsProvider

qulacs_provider = QulacsProvider()
```

You can then use this provider to get the available backends and execute your quantum circuits using Qulacs. Currently, only the statevector simulator is supported. Here's a basic example of how to use it:

```python
import qiskit
from qiskit_qulacs import QulacsProvider

# Create a quantum circuit
qc = qiskit.QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Use Qiskit-Qulacs to run the circuit
backend = QulacsProvider().get_backend('qulacs_simulator')
result = backend.run(qc).result()

# Get the statevector
statevector = result.get_statevector()

# Print the statevector
print(statevector)

# Output: [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
```

### QulacsEstimator
The QulacsEstimator is based on qiskit's estimator primitive but behind the scenes uses qulacs to compute the expectation values quickly. It allows you to estimate expectation values of observables using Qulacs. Here's a basic example of how to use it:

```python
import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp

from qiskit_qulacs.qulacs_estimator import QulacsEstimator

np.random.seed(42)

# Create a two-local quantum circuit with 3 qubits
qc = TwoLocal(
    3, ["ry", "rz", "rx"], ["cx"], "linear", 1, insert_barriers=False
).decompose()

# Generate random parameter values for the circuit
params = np.random.rand(qc.num_parameters)

# Create a SparsePauliOp observable
obs = SparsePauliOp.from_list([("Z" * qc.num_qubits, 0.5)])

# Initialize QulacsEstimator
qulacs_estimator = QulacsEstimator()

# Run the estimation job with the circuit, observable, and parameters
job = qulacs_estimator.run(qc, obs, params)

# Get the result of the job
result = job.result()

# Retrieve the expectation value from the result
expectation_value = result.values[0]

# Print the expectation value
print("Expectation value:", expectation_value)

# Output: Expectation value: -0.018581644467403284
```

### QulacsEstimatorGradient
The QulacsEstimatorGradient is based to qiskit's gradient framework. It allows you to compute gradients of quantum circuits using Qulacs. Here's a basic 1 qubit example.

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from qiskit_qulacs.qulacs_estimator_gradient import QulacsEstimatorGradient

# Create a two-local quantum circuit with 1 qubit with a parameter
qc = QuantumCircuit(1)

x = Parameter("x")
qc.rx(x, 0)

# Create a SparsePauliOp observable
obs = SparsePauliOp.from_list([("Z" * qc.num_qubits, 0.5)])

# Initialize QulacsEstimatorGradient
qulacs_grad = QulacsEstimatorGradient()

# Run the job with the circuit, observable, and parameters
job = qulacs_grad.run([qc], [obs], [[np.pi / 3]])

# Get the result of the job and retrieve the gradients
result = job.result()
gradient = result.gradients[0]

print("Gradient:", gradient)

# Output: Gradient: [-0.4330127]
```

## Conclusion
Congratulations! You have completed the Quickstart Guide for Qiskit-Qulacs. You should now have a basic understanding of the installation process and how to use the main components of Qiskit-Qulacs. Feel free to refer to the documentation and examples for more advanced usage and explore the possibilities of quantum computing with Qiskit-Qulacs. Happy coding!