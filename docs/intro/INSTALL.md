# Installation Guide

This document provides step-by-step instructions to set up the Python environment, install dependencies, and install the Qiskit-Qulacs plugin.

Setting up Python Environment
## Setting up Python Environment

Create a fresh Python environment using your preferred method (e.g., virtualenv, conda).

```
virtualenv qiskit-qulacs-env
source qiskit-qulacs-env/bin/activate
```

Ensure that you have a compatible version of Python (e.g., Python 3.7 or higher) installed in the environment.

## Installing Qiskit-Qulacs Software

### Start locally

The simplest way to get started is to use the pip package manager:

```
pip install qiskit-qulacs
```

### Install from source

Installing Qiskit-Qulacs from source allows you to access the most recently updated version under development, instead of using the version in the Python Package Index (PyPI) repository.

1. Clone the Qiskit Algorithms repository.
```
git clone https://github.com/Gopal-Dahale/qiskit-qulacs.git
```
2. Cloning the repository creates a local folder called ``qiskit-qulacs``.
```
cd qiskit-qulacs
```
3. Install ``qiskit-qulacs``.
```
pip install .
```
If you want to install it in editable mode, meaning that code changes to the project don't require a reinstall to be applied, you can do this with:
```
pip install -e .
```

## Testing the Installation

Since Qiskit-Qulacs is currently under heavy development, there are no tests yet. However, you can test the installation using a simple program:

```python
from qiskit import QuantumCircuit
from qiskit_qulacs import QulacsProvider

# Create a bell state
qc = QuantumCircuit(2)
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